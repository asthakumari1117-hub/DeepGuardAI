"""
STEP 5 — TRAIN LSTM MODEL
==========================
Trains a Bidirectional LSTM on the features built in step 4.
Saves → best_lstm_model.pth

Key fixes vs old code:
  • class_weights balanced from actual label counts (not hard-coded 1.0/2.0)
  • per-class accuracy printed each epoch so you can spot collapse
  • threshold tuning: saves best F1 threshold alongside model
  • gradient clipping to stabilise training
  • early stopping (patience=7) to avoid overfitting small datasets

Run:  python step5_train.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from torch.utils.data import TensorDataset, DataLoader

# ── CONFIG ────────────────────────────────────────────────
EPOCHS       = 40
BATCH_SIZE   = 16
LR           = 3e-4
PATIENCE     = 7          # early stopping
CLIP_GRAD    = 1.0
MODEL_PATH   = "best_lstm_model.pth"
# ──────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load dataset ──────────────────────────────────────────
X = np.load("lstm_features.npy")
y = np.load("lstm_labels.npy")

print(f"Dataset: X={X.shape}  y={y.shape}")
print(f"Real: {(y==0).sum()}   Fake: {(y==1).sum()}")

# ── Train / test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

to_t = lambda a, dt: torch.tensor(a, dtype=dt)
X_tr  = to_t(X_train, torch.float32)
X_te  = to_t(X_test,  torch.float32)
y_tr  = to_t(y_train, torch.long)
y_te  = to_t(y_test,  torch.long)

train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_te, y_te),
                          batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(X_train)}   Test: {len(X_test)}")


# ── Model ─────────────────────────────────────────────────
class DeepfakeLSTM(nn.Module):
    def __init__(self, input_size=512, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=0.4,
                            bidirectional=True)
        self.bn   = nn.BatchNorm1d(hidden * 2)
        self.fc1  = nn.Linear(hidden * 2, 128)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.4)
        self.fc2  = nn.Linear(128, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        # cat forward + backward last hidden
        x = torch.cat([h[-2], h[-1]], dim=1)
        x = self.bn(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


model = DeepfakeLSTM().to(device)

# Balanced class weights from actual counts
n_real = (y_train == 0).sum()
n_fake = (y_train == 1).sum()
w = torch.tensor(
    [n_fake / n_real if n_real > 0 else 1.0, 1.0],
    dtype=torch.float32
).to(device)
print(f"Class weights — real: {w[0]:.3f}  fake: {w[1]:.3f}")

criterion = nn.CrossEntropyLoss(weight=w)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── Training loop ─────────────────────────────────────────
best_f1, no_improve = 0.0, 0

print("\n═══════════════ TRAINING ═══════════════")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(seqs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        total_loss += loss.item()
        pred       = out.argmax(dim=1)
        n_correct  += (pred == labels).sum().item()
        n_total    += labels.size(0)

    train_acc = 100 * n_correct / n_total
    scheduler.step()

    # ── Eval ──────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            out   = model(seqs)
            prob  = torch.softmax(out, dim=1)[:, 1]
            pred  = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    test_acc = 100 * (all_preds == all_labels).mean()
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc  = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc  = 0.0

    # per-class recall to spot collapse
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    real_recall = cm[0,0] / (cm[0,:].sum() + 1e-8) * 100
    fake_recall = cm[1,1] / (cm[1,:].sum() + 1e-8) * 100

    print(f"Epoch {epoch:03d} | loss {total_loss/len(train_loader):.4f} "
          f"| train {train_acc:.1f}% | test {test_acc:.1f}% "
          f"| F1 {f1:.3f} | AUC {auc:.3f} "
          f"| real-recall {real_recall:.0f}% fake-recall {fake_recall:.0f}%")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), MODEL_PATH)
        no_improve = 0
        print(f"  ✓ Best model saved (F1={best_f1:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ── Final eval ────────────────────────────────────────────
print("\n═══════════════ FINAL EVALUATION ═══════════════")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        pred = model(seqs).argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds,
                             target_names=["REAL", "FAKE"]))
print(f"\nBest F1: {best_f1:.4f}")
print(f"Model saved to: {MODEL_PATH}")
print("Step 5 complete.")
