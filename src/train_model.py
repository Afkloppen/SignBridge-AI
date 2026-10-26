
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_FILE = Path("data/processed/landmarks.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "signbridge_lstm.pt"
LABELS_FILE = MODEL_DIR / "labels.json"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing feature file: {DATA_FILE}")

data = np.load(DATA_FILE, allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(str)
splits = data["splits"].astype(str)

# Keep only classes represented in the training split.
train_labels = sorted(set(y[splits == "train"]))
keep = np.isin(y, train_labels)
X, y, splits = X[keep], y[keep], splits[keep]

label_to_id = {label: i for i, label in enumerate(train_labels)}
y_ids = np.array([label_to_id[label] for label in y], dtype=np.int64)

train_mask = splits == "train"
val_mask = splits == "val"
test_mask = splits == "test"

if not train_mask.any() or not val_mask.any() or not test_mask.any():
    raise ValueError("Train, validation, and test data must all be present.")

# Normalize features using training data only.
mean = X[train_mask].mean(axis=(0, 1), keepdims=True)
std = X[train_mask].std(axis=(0, 1), keepdims=True)
std[std < 1e-6] = 1.0
X = (X - mean) / std

def make_loader(mask, shuffle=False):
    features = torch.tensor(X[mask], dtype=torch.float32)
    targets = torch.tensor(y_ids[mask], dtype=torch.long)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=16, shuffle=shuffle)

train_loader = make_loader(train_mask, shuffle=True)
val_loader = make_loader(val_mask)
test_loader = make_loader(test_mask)

class SignLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        last_output = sequence[:, -1, :]
        return self.classifier(self.dropout(last_output))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLSTM(X.shape[2], len(train_labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
best_state = None
patience = 10
epochs_without_improvement = 0
max_epochs = 60

print("Device:", device)
print("Videos:", len(X))
print("Classes:", len(train_labels), train_labels)
print("Split sizes:", {
    "train": int(train_mask.sum()),
    "val": int(val_mask.sum()),
    "test": int(test_mask.sum()),
})

# Train and select the best checkpoint using validation loss.
for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0.0

    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(targets)

    model.eval()
    val_loss = 0.0
    val_true, val_pred = [], []

    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            val_loss += loss.item() * len(targets)
            val_true.extend(targets.cpu().numpy())
            val_pred.extend(logits.argmax(dim=1).cpu().numpy())

    train_loss /= int(train_mask.sum())
    val_loss /= int(val_mask.sum())
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
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping.")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)
model.eval()

# Evaluate on the held-out test split only once after model selection.
test_true, test_pred = [], []
with torch.no_grad():
    for features, targets in test_loader:
        logits = model(features.to(device))
        test_true.extend(targets.numpy())
        test_pred.extend(logits.argmax(dim=1).cpu().numpy())

print("\n========== TEST RESULTS ==========")
print("Test accuracy:", round(accuracy_score(test_true, test_pred), 4))
print(classification_report(
    test_true,
    test_pred,
    labels=list(range(len(train_labels))),
    target_names=train_labels,
    zero_division=0,
))

# Save the model, label mapping, and normalization values for inference.
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": int(X.shape[2]),
        "num_classes": len(train_labels),
        "hidden_size": 64,
        "num_layers": 1,
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
print("Label mapping saved to:", LABELS_FILE)
