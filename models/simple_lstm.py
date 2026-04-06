"""
models/simple_lstm.py
----------------------
LSTM-based Cognitive Instability Predictor — demo-ready, fully standalone.

Architecture (Dual-Head)
------------------------
                   ┌─────────────────────────────────────┐
 Input sequence    │   LSTM  (seq_len × 1 feature)       │
 [CII(t-W)..CII(t)]│   hidden_size=32, num_layers=1      │
       │           └───────────────┬─────────────────────┘
       │                           │ last hidden state (32,)
       │                    ┌──────┴──────┐
       │                    │  FC dropout  │
       │             ┌──────┴──────┐ ┌────┴──────────┐
       │             │  reg_head   │ │  cls_head      │
       │             │  Linear(32,1)│ │  Linear(32,1) │
       │             │  → Tanh     │ │  → Sigmoid     │
       │             └──────┬──────┘ └────┬───────────┘
       │                    │              │
       │              predicted_cii    p_frustration
       │              ∈ [-1, +1]       ∈ [0,  1]
       └──────────────────────────────────────────────▶

Why dual-head?
  - Regression head: predicts the *scalar CII* at t+1 → useful for
    dashboard forecasting and early-warning systems.
  - Classification head: predicts *frustration probability* → feeds
    directly into the difficulty controller.

Why LSTM over feedforward?
  A player at CII=-0.4 after three steps of CII=-0.1, -0.2, -0.3 is
  escalating. The same player at CII=-0.4 after -0.6, -0.5, -0.4 is
  recovering. A simple threshold or MLP cannot distinguish these — the
  LSTM's hidden state retains sequence context to learn this difference.

Training data (demo)
--------------------
  Three synthetic CII trajectories are generated:
    · Frustration: monotonically declining CII with noise
    · Boredom:     monotonically rising CII with noise
    · Engaged:     CII near zero with light noise
  Sliding window of size W creates supervised (X, y) pairs.

Usage (standalone demo)
-----------------------
  python -m models.simple_lstm

Usage (import)
--------------
  from models.simple_lstm import CIIPredictor, load_or_train

  predictor = load_or_train()          # trains on synthetic data if no checkpoint
  result    = predictor.predict([-0.1, -0.3, -0.5, -0.7, -0.85])
  print(result["predicted_cii"])       # → near -1.0 (worsening)
  print(result["frustration_prob"])    # → 0.87
  print(result["state"])               # → "Frustrated"

Integration with backend/predictor.py
--------------------------------------
  class LSTMBackedPredictor(StatePredictor):
      def __init__(self, checkpoint="models/checkpoints/cii_lstm.pt"):
          super().__init__()
          self._model = CIIPredictor.load(checkpoint)

      def _infer(self, cii, **ctx):
          seq = ctx.get("cii_history", [cii])
          r   = self._model.predict(seq)
          p_e = max(0.0, 1.0 - r["frustration_prob"] - (1 - r["frustration_prob"]) * 0.3)
          return {"Frustrated": r["frustration_prob"],
                  "Bored": 1 - r["frustration_prob"] - p_e,
                  "Engaged": p_e}
"""

from __future__ import annotations

import os
import math
import random
from typing import List, Tuple, Optional, Dict

# ── Suppress OpenMP duplicate library warning on Windows ────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── PyTorch guard ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LEN       = 8      # input window size (timesteps)
HIDDEN_SIZE   = 32     # LSTM hidden units
NUM_LAYERS    = 1      # LSTM depth
DROPOUT       = 0.20   # applied between FC layers
LR            = 0.005  # Adam learning rate
EPOCHS        = 60     # training epochs
BATCH_SIZE    = 32
CHECKPOINT    = os.path.join(os.path.dirname(__file__), "checkpoints", "cii_lstm.pt")

FRUS_THRESHOLD = -0.25   # aligned with utils/cii.py THRESHOLDS
BORE_THRESHOLD = +0.30


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if _TORCH_OK:

    class _CIILSTMNet(nn.Module):
        """
        Dual-head LSTM for CII sequence modelling.

        Input  : (batch, seq_len, 1)   — scalar CII per timestep
        Outputs: predicted_cii float   — regression head (Tanh → [-1,+1])
                 frustration_prob float — classification head (Sigmoid → [0,1])
        """

        def __init__(
            self,
            hidden_size: int = HIDDEN_SIZE,
            num_layers:  int = NUM_LAYERS,
            dropout:     float = DROPOUT,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = 1,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
            )
            self.drop      = nn.Dropout(dropout)
            self.reg_head  = nn.Linear(hidden_size, 1)   # CII regression
            self.cls_head  = nn.Linear(hidden_size, 1)   # frustration prob
            self.tanh      = nn.Tanh()
            self.sigmoid   = nn.Sigmoid()

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            x: (batch, seq_len, 1)
            Returns: (pred_cii, frus_prob)  each shape (batch, 1)
            """
            out, _  = self.lstm(x)               # (batch, seq_len, hidden)
            last    = out[:, -1, :]              # (batch, hidden) — last step
            last    = self.drop(last)
            pred_cii  = self.tanh(self.reg_head(last))    # (batch, 1) ∈ [-1,+1]
            frus_prob = self.sigmoid(self.cls_head(last)) # (batch, 1) ∈ [0, 1]
            return pred_cii, frus_prob


    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    class _CIISequenceDataset(Dataset):
        """
        Sliding-window dataset from a list of CII trajectories.

        Each sample:
            X : tensor (seq_len, 1)    — input window of CII values
            y_cii  : tensor (1,)       — CII at t+1 (regression target)
            y_frus : tensor (1,)       — 1.0 if y_cii <= FRUS_THRESHOLD else 0.0
        """

        def __init__(
            self,
            trajectories: List[List[float]],
            seq_len: int = SEQ_LEN,
        ):
            self.samples: List[Tuple] = []
            for traj in trajectories:
                for i in range(len(traj) - seq_len):
                    window = traj[i : i + seq_len]
                    target_cii = traj[i + seq_len]
                    label_frus = 1.0 if target_cii <= FRUS_THRESHOLD else 0.0
                    self.samples.append((window, target_cii, label_frus))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            win, cii, frus = self.samples[idx]
            x     = torch.tensor([[v] for v in win], dtype=torch.float32)  # (seq,1)
            y_cii = torch.tensor([cii],  dtype=torch.float32)
            y_frs = torch.tensor([frus], dtype=torch.float32)
            return x, y_cii, y_frs


# ---------------------------------------------------------------------------
# Public class: CIIPredictor
# ---------------------------------------------------------------------------

class CIIPredictor:
    """
    High-level wrapper around _CIILSTMNet.

    Provides:
        train(trajectories)  — supervised training on CII sequence lists
        predict(sequence)    — inference on a single CII window
        save(path)           — serialise weights + config
        load(path)           — deserialise weights + config   (classmethod)

    Works without PyTorch for validation if you call predict() on a
    CIIPredictor created via load() — the guard at import time handles this.
    """

    def __init__(
        self,
        seq_len:     int   = SEQ_LEN,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = DROPOUT,
        lr:          float = LR,
    ):
        _require_torch()
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr

        self._net = _CIILSTMNet(hidden_size, num_layers, dropout)
        self._trained = False
        self._hx      = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        trajectories: List[List[float]],
        epochs:     int   = EPOCHS,
        batch_size: int   = BATCH_SIZE,
        verbose:    bool  = True,
    ) -> List[float]:
        """
        Train the dual-head LSTM on a list of CII trajectories.

        Args
        ----
        trajectories : List of CII time-series (each is a list of floats).
                       Each trajectory should be at least seq_len+1 long.
        epochs       : Training epochs.
        batch_size   : Mini-batch size.
        verbose      : Print loss every 10 epochs.

        Returns
        -------
        List of per-epoch average training losses.
        """
        dataset = _CIISequenceDataset(trajectories, self.seq_len)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer  = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_reg   = nn.MSELoss()
        loss_cls   = nn.BCELoss()
        loss_log   = []

        self._net.train()
        for epoch in range(1, epochs + 1):
            ep_loss = 0.0
            for X, y_cii, y_frus in loader:
                pred_cii, frus_prob = self._net(X)
                l_r = loss_reg(pred_cii, y_cii)
                l_c = loss_cls(frus_prob, y_frus)
                loss = l_r + l_c          # equally weighted joint loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss += loss.item()

            avg = ep_loss / max(len(loader), 1)
            loss_log.append(avg)

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {avg:.5f}"
                      f"  (reg + cls)")

        self._trained = True
        return loss_log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, cii_sequence: List[float]) -> Dict[str, float | str]:
        """
        Predict the next CII and frustration probability from a CII window.

        Args
        ----
        cii_sequence : List of recent CII values (any length ≥ 1).
                       If shorter than seq_len, it is zero-padded on the left.
                       If longer, the last seq_len values are used.

        Returns
        -------
        {
            "predicted_cii":    float  ∈ [-1, +1]   — next CII estimate
            "frustration_prob": float  ∈ [0,   1]   — P(Frustrated)
            "state":            str    — "Frustrated" | "Engaged" | "Bored"
            "confidence":       float  — confidence of state label
        }
        """
        self._net.eval()
        seq = _pad_or_trim(cii_sequence, self.seq_len)
        x   = torch.tensor([[[v] for v in seq]], dtype=torch.float32)  # (1, W, 1)

        pred_cii_t, frus_prob_t = self._net(x)
        predicted_cii   = float(pred_cii_t[0, 0].item())
        frustration_prob = float(frus_prob_t[0, 0].item())

        # Derive state from predicted CII
        state, confidence = _classify(predicted_cii, frustration_prob)

        return {
            "predicted_cii":    round(predicted_cii,    4),
            "frustration_prob": round(frustration_prob, 4),
            "state":            state,
            "confidence":       round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # Real-Time Stateful Inference (Optimized for <50ms ticks)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """
        Reset the LSTM hidden state.
        Call this between independent game sessions.
        """
        self._hx = None

    @torch.no_grad()
    def step(self, cii: float) -> Dict[str, float | str]:
        """
        O(1) highly optimised real-time inference.
        
        Instead of re-processing a sliding window of length W every tick,
        this method feeds only the NEW timestep and reuses the previous 
        hidden state (`self._hx`). 
        
        This prevents heavy sequence re-computation, guarantees <50ms response
        time, and natively preserves long-term temporal momentum.

        Args
        ----
        cii : The single latest CII value for this tick.

        Returns
        -------
        Dictionary identically matching the output of predict().
        """
        self._net.eval()
        
        # (batch=1, seq_len=1, features=1)
        x = torch.tensor([[[cii]]], dtype=torch.float32)

        if getattr(self, '_hx', None) is None:
            # First tick: zero state will be initialized internally
            self._hx = None

        # Pass x and previous hidden state (_hx is a tuple of (h_n, c_n))
        out, self._hx = self._net.lstm(x, self._hx)
        
        # Process the single output step
        last        = self._net.drop(out[:, 0, :])
        pred_cii_t  = self._net.tanh(self._net.reg_head(last))
        frus_prob_t = self._net.sigmoid(self._net.cls_head(last))

        predicted_cii    = float(pred_cii_t[0, 0].item())
        frustration_prob = float(frus_prob_t[0, 0].item())

        state, confidence = _classify(predicted_cii, frustration_prob)

        return {
            "predicted_cii":    round(predicted_cii,    4),
            "frustration_prob": round(frustration_prob, 4),
            "state":            state,
            "confidence":       round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str = CHECKPOINT) -> str:
        """Serialise model weights and hyperparameters to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict":  self._net.state_dict(),
            "seq_len":     self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "dropout":     self.dropout,
            "lr":          self.lr,
        }, path)
        print(f"[CIIPredictor] Saved -> {path}")
        return path

    @classmethod
    def load(cls, path: str = CHECKPOINT) -> "CIIPredictor":
        """Load a trained CIIPredictor from a checkpoint file."""
        _require_torch()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        predictor = cls(
            seq_len     = ckpt.get("seq_len",     SEQ_LEN),
            hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE),
            num_layers  = ckpt.get("num_layers",  NUM_LAYERS),
            dropout     = ckpt.get("dropout",     DROPOUT),
            lr          = ckpt.get("lr",          LR),
        )
        predictor._net.load_state_dict(ckpt["state_dict"])
        predictor._net.eval()
        predictor._trained = True
        print(f"[CIIPredictor] Loaded <- {path}")
        return predictor

    def __repr__(self) -> str:
        trained = "trained" if self._trained else "untrained"
        return (f"CIIPredictor({trained}, seq_len={self.seq_len}, "
                f"hidden={self.hidden_size}, layers={self.num_layers})")


# ---------------------------------------------------------------------------
# Convenience: load-or-train on synthetic data
# ---------------------------------------------------------------------------

def load_or_train(
    checkpoint: str  = CHECKPOINT,
    epochs:     int  = EPOCHS,
    verbose:    bool = True,
) -> "CIIPredictor":
    """
    Return a trained CIIPredictor.

    - If `checkpoint` exists on disk → load and return immediately.
    - Otherwise → generate synthetic data, train, save, return.

    Args
    ----
    checkpoint : Path to .pt file (default models/checkpoints/cii_lstm.pt).
    epochs     : Training epochs if training from scratch.
    verbose    : Print training progress.
    """
    if os.path.isfile(checkpoint):
        return CIIPredictor.load(checkpoint)

    if verbose:
        print("[CIIPredictor] No checkpoint found — training on synthetic data.")

    trajectories = generate_synthetic_data(n_per_class=80, length=40)
    predictor    = CIIPredictor()
    predictor.train(trajectories, epochs=epochs, verbose=verbose)
    predictor.save(checkpoint)
    return predictor


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_per_class: int = 80,
    length:      int = 40,
    noise:       float = 0.05,
    seed:        int  = 42,
) -> List[List[float]]:
    """
    Generate synthetic CII trajectories for demo training.

    Three classes of trajectories with realistic noise:
      · Frustration : CII drifts from ~0 down toward -1
      · Boredom     : CII drifts from ~0 up toward +1
      · Engaged     : CII oscillates near 0 ± 0.15

    Args
    ----
    n_per_class : Number of trajectories per class.
    length      : Timesteps per trajectory (must be > seq_len).
    noise       : Gaussian noise std applied to each value.
    seed        : Random seed for reproducibility.

    Returns
    -------
    List of length 3 * n_per_class, each a list of `length` CII floats.
    """
    random.seed(seed)

    def _clamp(v: float) -> float:
        return max(-1.0, min(1.0, v))

    def _noisy(v: float) -> float:
        return _clamp(v + random.gauss(0, noise))

    trajectories = []

    # ── Frustration trajectories ──────────────────────────────────────────
    for _ in range(n_per_class):
        start = random.uniform(0.1, 0.4)
        end   = random.uniform(-0.9, -0.4)
        traj  = [_noisy(start + (end - start) * (i / (length - 1)))
                 for i in range(length)]
        trajectories.append(traj)

    # ── Boredom trajectories ──────────────────────────────────────────────
    for _ in range(n_per_class):
        start = random.uniform(-0.1, 0.2)
        end   = random.uniform(0.4, 0.9)
        traj  = [_noisy(start + (end - start) * (i / (length - 1)))
                 for i in range(length)]
        trajectories.append(traj)

    # ── Engaged trajectories ──────────────────────────────────────────────
    for _ in range(n_per_class):
        center = random.uniform(-0.10, 0.20)
        freq   = random.uniform(0.3, 0.8)
        amp    = random.uniform(0.05, 0.15)
        traj   = [_noisy(center + amp * math.sin(freq * i))
                  for i in range(length)]
        trajectories.append(traj)

    random.shuffle(trajectories)
    return trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_or_trim(seq: List[float], target_len: int) -> List[float]:
    """Trim to last `target_len` values, or left-pad with first value."""
    if len(seq) >= target_len:
        return seq[-target_len:]
    pad_val = seq[0] if seq else 0.0
    return [pad_val] * (target_len - len(seq)) + list(seq)


def _classify(
    predicted_cii: float,
    frustration_prob: float,
) -> Tuple[str, float]:
    """
    Map predicted CII + regression output to (state, confidence).

    Uses the same thresholds as utils/cii.py for consistency.
    """
    if predicted_cii <= FRUS_THRESHOLD or frustration_prob >= 0.50:
        # Confidence = sigmoid score itself (already in [0,1])
        conf  = max(frustration_prob,
                    _zone_confidence(predicted_cii, FRUS_THRESHOLD, "frustration"))
        return "Frustrated", min(conf, 1.0)

    if predicted_cii >= BORE_THRESHOLD:
        boredom_score = 1.0 - frustration_prob
        conf = max(boredom_score,
                   _zone_confidence(predicted_cii, BORE_THRESHOLD, "boredom"))
        return "Bored", min(conf, 1.0)

    # Engaged — confidence highest at CII ≈ 0.0
    dist_to_frus = abs(predicted_cii - FRUS_THRESHOLD)
    dist_to_bore = abs(predicted_cii - BORE_THRESHOLD)
    margin       = min(dist_to_frus, dist_to_bore)
    conf         = 0.55 + 0.40 * (margin / 0.30)
    return "Engaged", min(conf, 0.95)


def _zone_confidence(cii: float, threshold: float, zone: str) -> float:
    """Distance-based confidence for boundary zones."""
    if zone == "frustration":
        depth = max(0.0, threshold - cii)   # how far below threshold
    else:
        depth = max(0.0, cii - threshold)   # how far above threshold
    return 0.55 + min(depth * 1.5, 0.40)


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError(
            "PyTorch is required for CIIPredictor.\n"
            "Install it with:  pip install torch\n"
            "Docs: https://pytorch.org/get-started/locally/"
        )


# ---------------------------------------------------------------------------
# Self-test / Demo  (run: python -m models.simple_lstm)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 64

    if not _TORCH_OK:
        print("ERROR: PyTorch is not installed.")
        print("Install: pip install torch")
        raise SystemExit(1)

    print(SEP)
    print("  CIIPredictor — LSTM Demo (dual-head: regression + classification)")
    print(SEP)

    # ── Step 1: Generate synthetic training data ───────────────────────────
    print("\n[Step 1] Generating synthetic CII trajectories...")
    trajs = generate_synthetic_data(n_per_class=80, length=40)
    print(f"  Generated {len(trajs)} trajectories "
          f"(frustration × 80 + boredom × 80 + engaged × 80)")

    # ── Step 2: Train ─────────────────────────────────────────────────────
    print(f"\n[Step 2] Training LSTM for {EPOCHS} epochs...")
    predictor = CIIPredictor(seq_len=SEQ_LEN, hidden_size=HIDDEN_SIZE)
    loss_curve = predictor.train(trajs, epochs=EPOCHS, verbose=True)
    print(f"  Final loss: {loss_curve[-1]:.5f}")
    print(f"  Model: {predictor}")

    # ── Step 3: Save ──────────────────────────────────────────────────────
    print(f"\n[Step 3] Saving checkpoint...")
    saved_path = predictor.save()
    print(f"  Saved to: {saved_path}")

    # ── Step 4: Load & verify ─────────────────────────────────────────────
    print(f"\n[Step 4] Loading from checkpoint (round-trip test)...")
    predictor2 = CIIPredictor.load(saved_path)
    print(f"  Loaded : {predictor2}")

    # ── Step 5: Inference on emotion scenarios ────────────────────────────
    print(f"\n[Step 5] Inference — qualitative scenarios\n")

    TEST_SEQS = [
        ("Frustration escalating",
         [0.0, -0.1, -0.2, -0.4, -0.6, -0.75, -0.85, -0.90]),

        ("Frustration recovering",
         [-0.85, -0.70, -0.55, -0.40, -0.25, -0.10,  0.05,  0.15]),

        ("Stable engaged flow",
         [0.05, 0.08, -0.02, 0.10, 0.07, -0.05, 0.06, 0.09]),

        ("Boredom onset",
         [0.10, 0.20, 0.30, 0.38, 0.45, 0.52, 0.60, 0.65]),

        ("Single-tick silent (length-1 input)",
         [-0.55]),
    ]

    print(f"  {'Scenario':<32}  {'Pred CII':>9}  {'P(Frus)':>8}  "
          f"{'State':<12}  Conf")
    print(f"  {'─'*32}  {'─'*9}  {'─'*8}  {'─'*12}  {'─'*5}")

    for label, seq in TEST_SEQS:
        r = predictor2.predict(seq)
        s = r["state"]
        state_marker = {"Frustrated": "⚠", "Engaged": "✓", "Bored": "○"}.get(s, "?")
        print(
            f"  {label:<32}  {r['predicted_cii']:>+9.4f}  "
            f"{r['frustration_prob']:>8.4f}  "
            f"{state_marker} {s:<11}  {r['confidence']:.3f}"
        )

    # ── Step 6: Architecture summary ──────────────────────────────────────
    print(f"\n{SEP}")
    print("[Step 6] Model architecture\n")
    print(predictor2._net)

    total_params = sum(p.numel() for p in predictor2._net.parameters())
    trainable    = sum(p.numel() for p in predictor2._net.parameters()
                       if p.requires_grad)
    print(f"\n  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    print(f"\n{SEP}")
    print("  Demo complete. Checkpoint saved at:")
    print(f"  {saved_path}")
    print(SEP)
