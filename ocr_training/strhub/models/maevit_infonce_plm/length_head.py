"""
Length Prediction Head for OCR models.

Predicts sequence length using encoder global features.
Uses classification approach instead of regression for stability.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class LengthHead(nn.Module):
    """
    Length prediction head using encoder global features.
    
    Args:
        embed_dim: Dimension of encoder output features
        max_length: Maximum sequence length to predict (num_classes = max_length + 1)
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, embed_dim: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        self.max_length = max_length
        self.num_classes = max_length + 1  # 0 to max_length (inclusive)
        
        # Simple MLP for length classification
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim // 2, self.num_classes)
        
    def forward(self, encoder_features: Tensor) -> Tensor:
        """
        Predict sequence length from encoder features.
        
        Args:
            encoder_features: Encoder output of shape (B, N, embed_dim)
                For MAE ViT: (B, 197, embed_dim) where first token is cls token
                
        Returns:
            logits: Length classification logits of shape (B, num_classes)
        """
        # Use cls token (first token) for global representation
        # Alternative: could use mean pooling over all tokens
        cls_token = encoder_features[:, 0, :]  # (B, embed_dim)
        
        x = self.fc1(cls_token)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (B, num_classes)
        
        return logits
    
    def predict(self, encoder_features: Tensor) -> Tensor:
        """
        Predict length (inference mode).
        
        Args:
            encoder_features: Encoder output
            
        Returns:
            predicted_lengths: Predicted sequence lengths of shape (B,)
        """
        logits = self.forward(encoder_features)
        predicted_lengths = logits.argmax(dim=-1)  # (B,)
        return predicted_lengths


def compute_length_loss(
    logits: Tensor, 
    targets: Tensor, 
    ignore_index: int = -100
) -> Tensor:
    """
    Compute cross-entropy loss for length prediction.
    
    Args:
        logits: Length prediction logits (B, num_classes)
        targets: Ground truth lengths (B,)
        ignore_index: Index to ignore in loss computation
        
    Returns:
        loss: Scalar loss value
    """
    return nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
