
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_FILE = Path("data/processed/landmarks.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "signbridge_cnn_lstm.pt"
LABELS_FILE = MODEL_DIR / "cnn_lstm_labels.json"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

data = np.load(DATA_FILE, allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(str)
splits = data["splits"].astype(str)

train_labels = sorted(set(y[splits == "train"]))
keep = np.isin(y, train_labels)
X, y, splits = X[keep], y[keep], splits[keep]

label_to_id = {label: i for i, label in enumerate(train_labels)}
y_ids = np.array([label_to_id[label] for label in y], dtype=np.int64)

train_mask = splits == "train"
val_mask = splits == "val"
test_mask = splits == "test"

if not train_mask.any() or not val_mask.any() or not test_mask.any():
    raise ValueError("Train, validation, and test splits must all exist.")

# Normalize using training data only.
mean = X[train_mask].mean(axis=(0, 1), keepdims=True)
std = X[train_mask].std(axis=(0, 1), keepdims=True)
std[std < 1e-6] = 1.0
X = (X - mean) / std

def make_loader(mask, shuffle=False):
    features = torch.tensor(X[mask], dtype=torch.float32)
    targets = torch.tensor(y_ids[mask], dtype=torch.long)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=16,
        shuffle=shuffle,
    )

train_loader = make_loader(train_mask, shuffle=True)
val_loader = make_loader(val_mask)
test_loader = make_loader(test_mask)

class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        # 1D CNN extracts local patterns from each frame's features.
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # LSTM learns temporal patterns across frames.
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: batch x frames x features
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.classifier(self.dropout(sequence[:, -1, :]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(X.shape[2], len(train_labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
best_state = None
patience = 10
stale_epochs = 0
max_epochs = 60

print("Device:", device)
print("Videos:", len(X))
print("Classes:", train_labels)
print("Split sizes:", {
    "train": int(train_mask.sum()),
    "val": int(val_mask.sum()),
    "test": int(test_mask.sum()),
})

for epoch in range(1, max_epochs + 1):
    model.train()
    total_train_loss = 0.0

    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * len(targets)

    model.eval()
    total_val_loss = 0.0
    val_true, val_pred = [], []

    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_val_loss += loss.item() * len(targets)
            val_true.extend(targets.cpu().numpy())
            val_pred.extend(logits.argmax(dim=1).cpu().numpy())

    train_loss = total_train_loss / int(train_mask.sum())
    val_loss = total_val_loss / int(val_mask.sum())
    val_accuracy = accuracy_score(val_true, val_pred)

    print(
        f"Epoch {epoch:02d}/{max_epochs} | "
        f"train loss={train_loss:.4f} | "
        f"val loss={val_loss:.4f} | "
        f"val accuracy={val_accuracy:.3f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= patience:
        print("Early stopping.")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a checkpoint.")

model.load_state_dict(best_state)
model.eval()

test_true, test_pred = [], []
with torch.no_grad():
    for features, targets in test_loader:
        logits = model(features.to(device))
        test_true.extend(targets.numpy())
        test_pred.extend(logits.argmax(dim=1).cpu().numpy())

print("\n========== CNN + LSTM TEST RESULTS ==========")
print("Test accuracy:", round(accuracy_score(test_true, test_pred), 4))
print(classification_report(
    test_true,
    test_pred,
    labels=list(range(len(train_labels))),
    target_names=train_labels,
    zero_division=0,
))

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": int(X.shape[2]),
        "num_classes": len(train_labels),
        "num_frames": int(X.shape[1]),
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    },
    MODEL_FILE,
)

LABELS_FILE.write_text(
    json.dumps(label_to_id, indent=2),
    encoding="utf-8",
)

print("\nModel saved to:", MODEL_FILE)
print("Labels saved to:", LABELS_FILE)
