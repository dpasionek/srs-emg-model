from __future__ import annotations

import torch
from torch import nn

# Using an Multilayer Perceptron (MLP) as a simple baseline model for EMG classification
# is probably our best bet here because it's:
# - Simple and fast to train
# - Can capture non-linear relationships in the data
# - Doesn't require a large amount of data to perform reasonably well

# NFT: You're welcome to play around with the "32" parameter here. This is the "width" of the hidden layer.
#   A larger number might capture more complex patterns but could also lead to overfitting,
#   especially if our dataset isn't very large. A smaller number might be more robust but could underfit. 
#   Starting with 32 is a common choice for a simple baseline, and we can adjust based on performance.

class BaselineEMGModel(nn.Module):
    def __init__(self, window_size: int, feature_count: int, num_classes: int) -> None:
        # Define a small MLP that flattens each window into a feature vector.
        super().__init__()
        input_dim = int(window_size) * int(feature_count)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run a forward pass to produce class logits for each window.
        return self.net(x)
