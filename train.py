"""Train the CKKS-compatible neural controller from the Webots dataset."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
N_INPUTS = 5
EPOCHS = 16
HIDDEN_DIM = 64
BATCH_SIZE = 1024
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-5

DATASET_PATH = Path(os.environ.get("AMR_DATASET", "webots_dataset.npz"))
MODEL_PATH = Path(os.environ.get("AMR_MODEL", "model_export.json"))


class OffsetSquareActivationMLP(nn.Module):
    """5 -> 64 -> 2 MLP with p(z) = z + 0.125 z^2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 2)

    @staticmethod
    def activation(x):
        return x + 0.125 * x * x

    def forward(self, x):
        return self.fc3(self.activation(self.fc1(x)))


def train_model(x_train, y_train, device):
    model = OffsetSquareActivationMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.train()
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 4 == 0 or epoch == 0 or epoch + 1 == EPOCHS:
            print(f"[Train] Epoch {epoch + 1}/{EPOCHS} complete")

    model.eval()
    return model, time.perf_counter() - start


def export_model(model, x_mean, x_std, y_mean, y_std):
    export = {
        "n_inputs": N_INPUTS,
        "feature_names": ["e_x", "e_y", "e_theta", "v_ref", "omega_ref"],
        # Transposed to [input][output] for NumPy and TenSEAL matmul.
        "fc1_weight": model.fc1.weight.detach().cpu().numpy().T.tolist(),
        "fc1_bias": model.fc1.bias.detach().cpu().numpy().tolist(),
        "fc3_weight": model.fc3.weight.detach().cpu().numpy().T.tolist(),
        "fc3_bias": model.fc3.bias.detach().cpu().numpy().tolist(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(export, f)


def main():
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET_PATH}. Run data_collector.py in Webots first."
        )

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    print(f"[Train] Loading dataset: {DATASET_PATH}")

    data = np.load(DATASET_PATH)
    x_raw = data["features"].astype(np.float64)
    y_raw = data["targets"].astype(np.float64)

    if x_raw.ndim != 2 or x_raw.shape[1] != N_INPUTS:
        raise SystemExit(
            f"Expected dataset features with shape (N, {N_INPUTS}), "
            f"but got {x_raw.shape}. Regenerate the dataset with the new collector."
        )

    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0) + 1e-6
    y_mean = y_raw.mean(axis=0)
    y_std = y_raw.std(axis=0) + 1e-6

    x_train = torch.tensor(
        (x_raw - x_mean) / x_std, dtype=torch.float32, device=device
    )
    y_train = torch.tensor(
        (y_raw - y_mean) / y_std, dtype=torch.float32, device=device
    )

    print(f"[Train] Dataset loaded: {len(x_raw)} samples. Starting training...")
    model, training_time = train_model(x_train, y_train, device)

    with torch.inference_mode():
        train_mse = ((model(x_train) - y_train) ** 2).mean().item()

    export_model(model, x_mean, x_std, y_mean, y_std)

    print(f"[Train] Samples: {len(x_raw)}")
    print(f"[Train] Training time: {training_time:.3f} s")
    print(f"[Train] Training MSE: {train_mse:.6f}")
    print(f"[Train] Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
