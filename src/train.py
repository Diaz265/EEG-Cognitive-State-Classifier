import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from load_eeg import load_eeg
from dataset import EEGDataset
from cnn_model import EEGCNN

# -------------------
# LOAD DATA
# -------------------
X, y = load_eeg()

# 🔥 NORMALISATION (CRUCIAL POUR 0.7)
X = (X - X.mean()) / (X.std() + 1e-8)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------
# CLASS BALANCING
# -------------------
class_counts = np.bincount(y)
weights = 1. / class_counts
sample_weights = weights[y]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# -------------------
# DATA
# -------------------
dataset = EEGDataset(X, y)
loader = DataLoader(dataset, batch_size=16, sampler=sampler)

# -------------------
# MODEL
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------
# TRAIN
# -------------------
for epoch in range(20):
    model.train()
    total_loss = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(Xb)

        loss = criterion(out, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# -------------------
# EVAL
# -------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        out = model(Xb)
        _, pred = torch.max(out, 1)

        total += yb.size(0)
        correct += (pred == yb).sum().item()

print("\nAccuracy:", correct / total)

torch.save(model.state_dict(), "eeg_model.pth")
print("Saved ✔")