
from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
from torch import nn

data = np.load("data/processed/landmarks.npz", allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(str)
splits = data["splits"].astype(str)

labels = sorted(set(y[splits == "train"]))
keep = np.isin(y, labels)
X, y, splits = X[keep], y[keep], splits[keep]

print("Dataset shape:", X.shape)
print("Labels:", labels)
print("Videos per label:", dict(Counter(y)))
print("Videos per split:", dict(Counter(splits)))

checkpoint = torch.load(
    "models/signbridge_cnn_lstm.pt",
    map_location="cpu",
    weights_only=True,
)

with open("models/cnn_lstm_labels.json", encoding="utf-8") as f:
    label_to_id = json.load(f)

id_to_label = {int(v): k for k, v in label_to_id.items()}

class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.classifier(self.dropout(sequence[:, -1, :]))

model = CNNLSTM(
    checkpoint["input_size"],
    checkpoint["num_classes"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

mean = np.array(checkpoint["mean"], dtype=np.float32).reshape(1, 1, -1)
std = np.array(checkpoint["std"], dtype=np.float32).reshape(1, 1, -1)
std[std < 1e-6] = 1.0

for split in ("train", "val", "test"):
    mask = splits == split
    if not mask.any():
        continue

    batch = (X[mask] - mean) / std
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(batch, dtype=torch.float32)), dim=1
        )
        pred_ids = probs.argmax(dim=1).numpy()

    predicted_labels = [id_to_label[int(i)] for i in pred_ids]
    actual_labels = y[mask]

    print(f"\n{split.upper()}:")
    print("Actual:", dict(Counter(actual_labels)))
    print("Predicted:", dict(Counter(predicted_labels)))
    print(
        "Correct:",
        sum(a == b for a, b in zip(actual_labels, predicted_labels)),
        "/",
        len(actual_labels),
    )
