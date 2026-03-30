"""
models/simple_lstm.py
---------------------
Simple LSTM for Temporal CII Prediction.

EXPLANATION:
Why an LSTM (Long Short-Term Memory)?
A standard Feed-Forward neural network (or rule-based trigger) only looks at the 
"current snapshot" of a player (e.g., CII = -0.4). This ignores **momentum**. 
A player going from 0.0 -> -0.2 -> -0.4 is escalating into frustration rapidly. 
A player hovering at [-0.4 -> -0.45 -> -0.4] might just be in a tough boss fight 
and successfully regulating their emotion.

The LSTM excels at finding patterns over sequences. In our scenario:
- Input Sequence: The last N values of the Cognitive Instability Index (CII).
- Target Output: The future instability value OR probability of frustration 
                 in the next T steps.

By feeding a sequence `[CII(t-4), CII(t-3), CII(t-2), CII(t-1), CII(t)]` into the LSTM,
the internal hidden state retains a "memory" of the emotional trajectory. It learns 
the difference between temporary setbacks and systemic frustration onset, effectively 
predicting the player's state *before* they actually quit.

Architecture Details:
- Input: Sequence length W (e.g., 5-10 timesteps), Feature dimension 1 (just CII).
- Hidden layer: LSTM cells track the sequence.
- Output: A Linear layer that emits the predicted future state (Regression or Classification).

Demo Script:
This basic implementation creates a PyTorch LSTM that takes a sequence of just 
CII values and predicts the 'future player state probability' [p_frustrated, p_bored].
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from typing import List

class SimpleCIILSTM(nn.Module):
    """
    Minimal LSTM model designed specifically for demo and easy integration.
    Only takes 1 feature per timestep: the CII scalar.
    Outputs 2 probabilities: [Frustration Probability, Boredom Probability].
    """
    
    def __init__(self, hidden_size: int = 16, num_layers: int = 1):
        super(SimpleCIILSTM, self).__init__()
        
        # input_size = 1 because we only feed in the scalar CII score
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer maps LSTM hidden state to our 2 target classes
        self.fc = nn.Linear(hidden_size, 2)
        
        # Maps outputs to probabilities between 0 and 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x is a tensor of shape (batch_size, sequence_length, features)
            Here features = 1 (just CII).
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # We only care about the prediction at the very last timestep of the sequence
        last_timestep_out = lstm_out[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Pass through fully connected layer and sigmoid
        logits = self.fc(last_timestep_out)     # Shape: (batch, 2)
        probs = self.sigmoid(logits)            # Shape: (batch, 2)
        
        return probs

    @torch.no_grad()
    def predict(self, cii_sequence: List[float]) -> dict:
        """
        Predict future state purely from a list of past CII values.
        Example: predictor.predict([-0.1, -0.3, -0.5, -0.7])
        -> {"frustrated": 0.85, "bored": 0.05}
        """
        self.eval() # set to evaluation mode
        
        # Prepare input: turn list into tensor of shape (1, seq_len, 1)
        # 1   : batch size 1
        # len : sequence length
        # 1   : 1 feature (CII)
        x = torch.tensor([[ [val] for val in cii_sequence ]], dtype=torch.float32)
        
        probs = self.forward(x)
        
        # Extract predicted probabilities
        p_frus = probs[0, 0].item()
        p_bore = probs[0, 1].item()
        
        return {
            "frustrated_prob": round(p_frus, 4),
            "bored_prob": round(p_bore, 4),
            "predicted_state": "frustrated" if p_frus > 0.5 else ("bored" if p_bore > 0.5 else "engaged")
        }


# ---------------------------------------------------------------------------
# Simple Demo Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Simple CII LSTM Model...")
    
    # 1. Initialize model
    model = SimpleCIILSTM(hidden_size=16)
    
    # 2. In a real-world scenario, you would train the model.
    # We will simulate predictions with the un-trained weights just to verify functionality.
    
    # Scenario A: Worsening Frustration
    # Notice the mathematical pattern: the CII is dropping deeper into the negatives.
    frust_seq = [0.0, -0.2, -0.5, -0.8]
    res_a = model.predict(frust_seq)
    
    print("\nSequence A (Escalating Frustration):", frust_seq)
    print("Prediction:", res_a)
    
    # Scenario B: Stagnant Boredom
    # CII is hovering firmly in positive territory
    bored_seq = [0.4, 0.45, 0.5, 0.55]
    res_b = model.predict(bored_seq)
    
    print("\nSequence B (Growing Boredom):", bored_seq)
    print("Prediction:", res_b) 

    print("\nArchitecture Summary:")
    print(model)
