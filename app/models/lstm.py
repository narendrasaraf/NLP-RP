"""
app/models/lstm.py
------------------
PyTorch LSTM model for temporal CII sequence prediction.

Architecture:
  Input  : (batch, seq_len=10, input_dim=5)  -- [CII, M, A, D, G] per step
  LSTM   : 2 layers, hidden=64, dropout=0.3
  Output : (batch, 2)  -- [p_frus, p_bore] via Sigmoid

Usage:
  Training:
      trainer = LSTMTrainer(model, optimizer, criterion)
      trainer.train(dataloader, epochs=50)
      trainer.save("checkpoints/lstm.pt")

  Inference:
      model = CIILSTMPredictor.load("checkpoints/lstm.pt")
      p_frus, p_bore = model.predict_single(feature_sequence)

Note: PyTorch is an optional dependency. If not installed, the system
falls back to SimulatedLSTMPredictor in app/core/predictor.py.
"""

# Guard against missing PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import os
from typing import List, Tuple, Optional
from app.core.config import LSTMConfig, settings


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class CIILSTMPredictor(nn.Module):
        """
        Two-layer LSTM classifier for cognitive state prediction.

        Input features per timestep (input_dim=5):
            [CII(t), M(t), A(t), D(t), G(t)]

        Outputs frustration and boredom probabilities independently
        via two Sigmoid units — treated as a multi-label binary problem.
        """

        def __init__(self, cfg: LSTMConfig = None):
            super().__init__()
            cfg = cfg or settings.lstm

            self.cfg = cfg
            self.lstm = nn.LSTM(
                input_size  = cfg.input_dim,
                hidden_size = cfg.hidden_dim,
                num_layers  = cfg.num_layers,
                batch_first = True,
                dropout     = cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.head    = nn.Linear(cfg.hidden_dim, 2)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            x: (batch, seq_len, input_dim)
            returns: (batch, 2)  -- [p_frus, p_bore]
            """
            lstm_out, _ = self.lstm(x)               # (batch, seq_len, hidden)
            last_step   = lstm_out[:, -1, :]          # (batch, hidden)
            out = self.dropout(last_step)
            out = self.head(out)                      # (batch, 2)
            return self.sigmoid(out)

        def predict_single(
            self,
            feature_sequence: List[List[float]],
        ) -> Tuple[float, float]:
            """
            Predict from a single sequence (no batch dim).

            Args:
                feature_sequence: list of W vectors, each [CII, M, A, D, G]

            Returns:
                (p_frus, p_bore)
            """
            self.eval()
            with torch.no_grad():
                x = torch.tensor(
                    [feature_sequence], dtype=torch.float32
                )  # (1, W, 5)
                out = self.forward(x)   # (1, 2)
                p_frus = out[0, 0].item()
                p_bore = out[0, 1].item()
            return round(p_frus, 4), round(p_bore, 4)

        def save(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)
            print(f"[LSTM] Model saved to {path}")

        @classmethod
        def load(cls, path: str) -> "CIILSTMPredictor":
            checkpoint = torch.load(path, map_location="cpu")
            cfg   = checkpoint["cfg"]
            model = cls(cfg)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            print(f"[LSTM] Model loaded from {path}")
            return model


    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    class CIIDataset(Dataset):
        """
        Sliding-window dataset for LSTM training.

        Each sample:
            X : (seq_len, input_dim)  -- sequence of [CII, M, A, D, G]
            y : (2,)                  -- [label_frus, label_bore]
        """

        def __init__(
            self,
            sequences: List[List[List[float]]],
            labels:    List[List[float]],
        ):
            assert len(sequences) == len(labels)
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
            self.labels    = torch.tensor(labels,    dtype=torch.float32)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]


    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------

    class LSTMTrainer:
        """
        Training loop for CIILSTMPredictor.

        Loss: Binary Cross Entropy (multi-label)
        Optimizer: Adam

        Example:
            model   = CIILSTMPredictor()
            trainer = LSTMTrainer(model)
            trainer.train(dataloader, epochs=50)
            trainer.save("checkpoints/lstm.pt")
        """

        def __init__(self, model: CIILSTMPredictor, cfg: LSTMConfig = None):
            self.model = model
            cfg = cfg or settings.lstm
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            self.criterion = nn.BCELoss()

        def train(
            self,
            dataloader: "DataLoader",
            epochs: int = 50,
            verbose: bool = True,
        ) -> List[float]:
            """Train for given epochs. Returns list of epoch losses."""
            self.model.train()
            losses = []
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                for X_batch, y_batch in dataloader:
                    preds = self.model(X_batch)
                    loss  = self.criterion(preds, y_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                avg = epoch_loss / len(dataloader)
                losses.append(avg)
                if verbose and epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {avg:.4f}")
            return losses

        def save(self, path: str):
            self.model.save(path)


else:
    # Stub classes when PyTorch is not installed
    class CIILSTMPredictor:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not installed. Run: pip install torch\n"
                "The system will use SimulatedLSTMPredictor as fallback."
            )

    class LSTMTrainer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMTrainer.")
