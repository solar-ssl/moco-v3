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
    Hybrid mode: supports memory queue for small batch training.
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=0.2, m=0.99, K=65536):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 0.2)
        m: moco momentum of updating key encoder (default: 0.99)
        K: queue size; number of negative samples (default: 65536)
        """
        super(MoCoV3, self).__init__()

        self.T = T
        self.m = m
        self.K = K

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
            nn.Linear(mlp_dim, dim, bias=False),
            nn.BatchNorm1d(dim)
        )

        # Build momentum encoder
        self.base_model_k = copy.deepcopy(self.base_model)
        self.projector_k = copy.deepcopy(self.projector_q)

        # Predictor (2-layer MLP)
        self.predictor = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim, bias=False),
            nn.BatchNorm1d(dim)
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

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.base_model.parameters(), self.base_model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue (if using DDP)
        # keys = concat_all_gather(keys) # Handled externally or assuming single-node logic for now

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at index 'ptr' (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # handle wrap-around
            rem = self.K - ptr
            self.queue[:, ptr:] = keys[:rem].T
            self.queue[:, :batch_size - rem] = keys[rem:].T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, use_queue=False):
        # Normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        if use_queue:
            # Positive logits: dot product of query and its corresponding key (Nx1)
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # Negative logits from queue (NxK)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            # Positive is at index 0
            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        else:
            # Standard MoCo v3 (batch as negatives)
            # NxN matrix where diagonal elements are positives
            logits = torch.einsum('nc,mc->nm', [q, k])
            labels = torch.arange(logits.shape[0], dtype=torch.long).to(logits.device)
        
        logits /= self.T
        return nn.CrossEntropyLoss()(logits, labels)

    def forward(self, x1, x2, m, use_queue=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: momentum value
            use_queue: whether to use memory queue for negatives
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
        loss = self.contrastive_loss(q1, k2, use_queue=use_queue) + \
               self.contrastive_loss(q2, k1, use_queue=use_queue)
        
        if use_queue:
            self._dequeue_and_enqueue(torch.cat([k1, k2], dim=0))

        return loss
