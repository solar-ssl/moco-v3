"""
MoCo v3 implementation.
Features two encoders (query and momentum key), projection head, and prediction head.

Distributed training note
--------------------------
In multi-GPU training each process only holds a local slice of the batch.
Without all_gather the contrastive loss only sees (local_batch_size - 1)
negatives per query instead of (total_batch_size - 1).  With 4 GPUs and
batch_size=256 that is 63 negatives instead of 255 — a 4× weaker signal.

The fix (concat_all_gather) gathers key embeddings from every GPU into one
large tensor before building the similarity matrix, while keeping gradients
flowing only through the local queries.  This matches the official MoCo v3
implementation exactly.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import copy

@torch.no_grad()
def concat_all_gather(tensor):
    """
    No-gradient all-gather used for key embeddings.

    Falls back gracefully to a no-op when not in a distributed context
    (single-GPU / CPU training), so the model works identically in both modes.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


class MoCoV3(nn.Module):
    """
    MoCo v3: momentum contrast with prediction head and momentum update.
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=0.2, m=0.99):
        """
        Args:
            base_encoder: callable() → (model, dim_in)
            dim:          final embedding dimension (default: 256)
            mlp_dim:      hidden dimension in MLPs (default: 4096)
            T:            softmax temperature (default: 0.2)
            m:            initial MoCo momentum; overridden per-step by the
                          cosine schedule in train_moco.py (default: 0.99)
        """
        super(MoCoV3, self).__init__()

        self.T = T

        self.base_model, dim_in = base_encoder()

        # Query projector: 3-layer MLP  dim_in → mlp_dim → mlp_dim → dim
        self.projector_q = nn.Sequential(
            nn.Linear(dim_in, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
        )

        # Momentum (key) encoder — deep-copied from query encoder
        self.base_model_k  = copy.deepcopy(self.base_model)
        self.projector_k   = copy.deepcopy(self.projector_q)

        # Predictor: 2-layer MLP  dim → mlp_dim → dim
        self.predictor = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
        )

        # Freeze momentum encoder — updated only via EMA, never by gradients
        for param_k in self.base_model_k.parameters():
            param_k.requires_grad = False
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False

        # Put the momentum encoder into eval mode immediately after construction.
        # train() is overridden below to keep it there permanently.
        self.base_model_k.eval()
        self.projector_k.eval()

    def train(self, mode: bool = True) -> "MoCoV3":
        """
        Override nn.Module.train() so that the momentum (key) encoder is
        *always* kept in eval mode, regardless of what the caller does.

        Why this matters
        ----------------
        nn.Module.train(True) recurses into every child module and flips all
        of them to training mode.  That includes base_model_k and projector_k,
        whose BatchNorm layers then switch from using stable running statistics
        (eval mode) to noisy per-batch statistics (train mode).

        Because the key encoder is called under torch.no_grad() and updated
        only via EMA, its BatchNorm running stats are never updated during the
        forward pass — the batch statistics it sees are therefore random,
        uncorrelated noise that destabilises the key embeddings and degrades
        contrastive loss quality.

        By overriding train() here the invariant is enforced at the model
        level: the caller (train_moco.py) can freely call model.train() without
        needing to know about this internal detail.
        """
        super().train(mode)
        # Always revert the momentum encoder to eval, unconditionally
        self.base_model_k.eval()
        self.projector_k.eval()
        return self

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """
        EMA update for learnable parameters AND BatchNorm running statistics.

        Parameters (weight, bias):
            key = m * key + (1 - m) * query

        Buffers (running_mean, running_var — dtype float):
            Same EMA rule, so the key encoder's BN normalisation tracks the
            query encoder's as it learns, rather than staying frozen at the
            values copied at construction time.
            num_batches_tracked (dtype long) is skipped — it is a counter,
            not a statistic, and EMA on integers is meaningless.
        """
        for pq, pk in zip(self.base_model.parameters(), self.base_model_k.parameters()):
            pk.data = pk.data * m + pq.data * (1. - m)
        for pq, pk in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            pk.data = pk.data * m + pq.data * (1. - m)
        # EMA the float buffers (running_mean / running_var); skip long counters
        for bq, bk in zip(self.base_model.buffers(), self.base_model_k.buffers()):
            if bq.dtype == torch.float32:
                bk.data = bk.data * m + bq.data * (1. - m)
        for bq, bk in zip(self.projector_q.buffers(), self.projector_k.buffers()):
            if bq.dtype == torch.float32:
                bk.data = bk.data * m + bq.data * (1. - m)
    
    def contrastive_loss(self, q, k):
        """
        InfoNCE loss: local queries against globally-gathered keys.

        Steps
        -----
        1. Normalise q and k to the unit hypersphere.
        2. Only gather keys globally (k_all, no grad). q stays local so
           gradients flow naturally — no custom autograd Function needed,
           and the loss magnitude does not scale with world_size.
        3. Build the [local_N × global_N] similarity matrix.
        4. The positive for rank r, local index i is at global column
           (r * local_N + i) — this rank's diagonal slice in k_all.

        Single-GPU / non-distributed: concat_all_gather is a no-op,
        rank=0, so labels are simply torch.arange(local_N).

        Args:
            q: [local_N, dim]  query embeddings  (gradients required)
            k: [local_N, dim]  key embeddings    (no grad)

        Returns:
            scalar loss
        """
        local_n = q.shape[0]

        q = nn.functional.normalize(q, dim=1)  # [local_N, dim]
        k = nn.functional.normalize(k, dim=1)  # [local_N, dim]

        # Gather keys from all GPUs; q stays local so gradients flow without
        # a custom autograd Function and the loss magnitude stays rank-independent.
        k_all = concat_all_gather(k)  # [global_N, dim], no grad

        # [local_N, global_N] similarity matrix
        logits = torch.einsum('nc,mc->nm', q, k_all) / self.T

        # The positive pair for rank r, local index i is at global column
        # (r * local_N + i) — this rank's diagonal slice in k_all.
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        labels = torch.arange(local_n, dtype=torch.long, device=q.device) + rank * local_n

        loss = nn.CrossEntropyLoss()(logits, labels)
        # The 2*T factor matches the official MoCo v3 implementation and keeps
        # loss magnitude proportional to temperature so LR=1.5e-4 stays correct.
        return loss * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Args:
            x1: view-1 batch  [N, C, H, W]
            x2: view-2 batch  [N, C, H, W]
            m:  current MoCo momentum value (cosine-scheduled by trainer)

        Returns:
            scalar contrastive loss
        """
        # Step 1: EMA-update the momentum encoder before computing features
        self._update_momentum_encoder(m)

        # Step 2: Query features  (query encoder + predictor, gradients on)
        q1 = self.predictor(self.projector_q(self.base_model(x1)))  # [N, dim]
        q2 = self.predictor(self.projector_q(self.base_model(x2)))  # [N, dim]

        # Step 3: Key features  (momentum encoder only, no gradients)
        with torch.no_grad():
            k1 = self.projector_k(self.base_model_k(x1))  # [N, dim]
            k2 = self.projector_k(self.base_model_k(x2))  # [N, dim]

        # Step 4: Symmetric loss — each view's query predicts the other's key
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return loss
