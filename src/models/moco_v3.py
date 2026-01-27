"""
MoCo v3 implementation.
Features two encoders (query and momentum key), projection head, and prediction head.
"""

import torch
import torch.nn as nn
import copy

class MoCoV3(nn.Module):
    """
    MoCo v3: momentum contrast with prediction head and momentum update.
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=0.2, m=0.99):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 0.2)
        m: moco momentum of updating key encoder (default: 0.99)
        """
        super(MoCoV3, self).__init__()

        self.T = T
        self.m = m

        # Create the encoders
        # base_encoder returns (model, dim_in)
        self.base_model, dim_in = base_encoder()
        
        # Projector for query encoder (3-layer MLP)
        self.projector_q = nn.Sequential(
            nn.Linear(dim_in, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )

        # Build momentum encoder
        self.base_model_k = copy.deepcopy(self.base_model)
        self.projector_k = copy.deepcopy(self.projector_q)

        # Predictor (2-layer MLP)
        self.predictor = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )

        # Deactivate gradient for momentum encoder
        for param_k in self.base_model_k.parameters():
            param_k.requires_grad = False
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False

        # Initialize momentum encoder with query encoder weights
        for param_q, param_k in zip(self.base_model.parameters(), self.base_model_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.base_model.parameters(), self.base_model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def contrastive_loss(self, q, k):
        # Normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # Einstein sum is more efficient for multi-gpu
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        labels = torch.arange(logits.shape[0], dtype=torch.long).to(logits.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: momentum value
        Output:
            loss
        """
        # Update momentum encoder
        self._update_momentum_encoder(m)

        # Compute query features
        q1 = self.predictor(self.projector_q(self.base_model(x1)))
        q2 = self.predictor(self.projector_q(self.base_model(x2)))

        # Compute key features
        with torch.no_grad():
            k1 = self.projector_k(self.base_model_k(x1))
            k2 = self.projector_k(self.base_model_k(x2))

        # Loss is symmetric
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return loss
