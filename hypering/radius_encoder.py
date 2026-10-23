"""
Distribution-aware Radius Encoder (design B).

`LearnableRadii` treats the K ring radii as free scalars that only ever see the
class MEAN radius (through L_align). This module instead *encodes* the current
radial distribution of every class and decides the ring layout from it, so the
spacing between two stages can depend on how much their radial distributions
spread and overlap, not just on where their centres sit.

    per-class radial statistics s_c          (detached, EMA across batches)
        │   s_c = [ q10, q25, q50, q75, q90, std, log π_c ]   of  ρ(e_i), y_i = c
        ▼
    token  t_c = W_s · LN(s_c) + E_stage[c]   (E_stage = learnable ordinal embedding)
        ▼
    Transformer over the K stage tokens       (stages attend to each other →
        │                                       neighbouring-stage overlap)
        ▼  h_0 … h_{K-1}
    gap logits  ℓ_t = MLP([h_t ‖ h_{t+1}])    t = 0 … K-2   (one per adjacent pair)
    proportions p = softmax(ℓ)                (simplex: gaps share one span)
    gaps        Δ_t = min_gap + S · p_t       (S = span excess, learnable or fixed)
    k_0         = floor + softplus(w·h_0 + b)
    radii       k_c = k_0 + Σ_{t<c} Δ_t       (strictly increasing by construction)

Properties kept from LearnableRadii
  • monotonicity and the min_gap / floor lower bounds are structural;
  • at initialisation the output is exactly the old uniform layout
    (last layers zero-initialised → p uniform, k_0 = init_spacing), so the
    first epochs behave like the old model and the freeze warm-up still works;
  • the statistics are DETACHED: the radius encoder reads the embeddings but
    the encoder cannot steer the rings by gaming those statistics. Gradients
    reach the radius encoder only through the radii, from L_align (the
    training scripts detach the radii in L_emb).
  • the same `alignment_loss` / `ceiling_loss` API, so the training scripts
    call it exactly like LearnableRadii.

Geometry-agnostic: pass `norm_fn` = hyperbolic ρ(·) or the euclidean ‖·‖, so
the hyperbolic and euclidean pipelines share one implementation.

Ablation note: this module can degenerate to "output the class median" — the
radius-encoder ablation (LearnableRadii vs RadiusEncoder) is what shows it
learned more than that.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
N_STATS = len(QUANTILES) + 2          # quantiles + std + log class prior


def _softplus_inv(x):
    x = max(float(x), 1e-4)
    return x if x > 20.0 else math.log(math.expm1(x))


class RadiusEncoder(nn.Module):
    """Set/transformer encoder that maps per-class radial distributions to radii.

    Args:
        num_classes:   K ordinal stages.
        norm_fn:       callable features -> ρ (N,). Bind the curvature in it for
                       the hyperbolic case (e.g. functools.partial).
        init_spacing:  initial uniform gap and initial k_0 (matches LearnableRadii).
        min_gap:       structural lower bound on every gap.
        floor:         structural lower bound on k_0.
        span:          None → total gap budget is learnable (old behaviour: the
                       rings may expand). float → fixed total span
                       k_{K-1} − k_0 = span; only the proportions are learned
                       (removes the overall-scale degree of freedom).
        ceiling, ceiling_weight: soft cap on the outermost ring (API parity).
        trim_ratio:    trimmed-mean fraction for alignment_loss (as before).
        d_model, n_heads, n_layers, dropout: stage-token transformer size.
        momentum:      EMA momentum for the per-class statistics across batches.
        init_props:    None → the initial gaps are equal (default). A length-K-1
                       sequence of positive numbers → the initial gap budget is
                       split in these proportions instead (normalised to sum 1);
                       used by the ring-layout ablation. Applied as a fixed
                       log-proportion offset on the gap logits, so the encoder
                       can still learn away from it.
    """

    def __init__(self, num_classes, norm_fn, init_spacing=0.5, min_gap=0.1,
                 floor=0.1, span=None, ceiling=3.0, ceiling_weight=1.0,
                 trim_ratio=0.1, d_model=32, n_heads=2, n_layers=2,
                 dropout=0.0, momentum=0.9, init_props=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError("RadiusEncoder needs at least 2 classes")
        K = num_classes
        self.num_classes = K
        self.norm_fn = norm_fn
        self.min_gap = float(min_gap)
        self.floor = float(floor)
        self.ceiling = ceiling
        self.ceiling_weight = ceiling_weight
        self.trim_ratio = float(trim_ratio)
        self.momentum = float(momentum)
        self.fixed_span = span is not None

        # running per-class radial statistics (saved in state_dict, so a
        # restored checkpoint reproduces its radii exactly)
        self.register_buffer('stats', torch.zeros(K, N_STATS))
        self.register_buffer('seen', torch.zeros(K, dtype=torch.bool))

        # stage tokens
        self.token_proj = nn.Sequential(nn.LayerNorm(N_STATS),
                                        nn.Linear(N_STATS, d_model))
        self.stage_emb = nn.Embedding(K, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # heads
        self.gap_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))
        self.k0_head = nn.Linear(d_model, 1)
        # identity start: uniform proportions, k_0 = init_spacing
        nn.init.zeros_(self.gap_head[-1].weight)
        nn.init.zeros_(self.gap_head[-1].bias)
        nn.init.zeros_(self.k0_head.weight)
        with torch.no_grad():
            self.k0_head.bias.fill_(_softplus_inv(max(init_spacing - floor, 0.01)))

        # initial split of the gap budget (zeros → equal gaps)
        if init_props is None:
            offset = torch.zeros(K - 1)
        else:
            p = torch.as_tensor(init_props, dtype=torch.float32).flatten()
            if p.numel() != K - 1 or bool((p <= 0).any()):
                raise ValueError(f"init_props must be {K - 1} positive numbers, "
                                 f"got {init_props}")
            offset = torch.log(p / p.sum())
        self.register_buffer('gap_offset', offset)

        # total gap budget S (excess over the (K-1)·min_gap floor)
        if self.fixed_span:
            excess = float(span) - (K - 1) * self.min_gap
            if excess <= 0:
                raise ValueError(f"span={span} must exceed (K-1)*min_gap="
                                 f"{(K - 1) * self.min_gap}")
            self.register_buffer('span_excess', torch.tensor(excess))
        else:
            excess0 = (K - 1) * max(init_spacing - self.min_gap, 0.01)
            self.span_raw = nn.Parameter(torch.tensor(_softplus_inv(excess0)))

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def observe(self, features, labels, momentum=None):
        """Update the per-class radial statistics from (detached) embeddings.

        momentum=None → EMA with self.momentum (per batch);
        momentum=0.0  → overwrite with these embeddings (use with the full
                        training set once per epoch for a stable estimate).
        Classes absent from `labels` keep their previous statistics.
        """
        m = self.momentum if momentum is None else float(momentum)
        rho = self.norm_fn(features.detach()).float()
        n = rho.numel()
        if n == 0:
            return
        q = torch.tensor(QUANTILES, device=rho.device, dtype=rho.dtype)
        for k in range(self.num_classes):
            mask = labels == k
            nk = int(mask.sum())
            if nk == 0:
                continue
            r = rho[mask]
            s = torch.cat([
                torch.quantile(r, q),
                r.std(unbiased=False).reshape(1),
                torch.tensor([math.log(nk / n)], device=rho.device, dtype=rho.dtype),
            ]).to(self.stats.dtype)
            if bool(self.seen[k]) and m > 0:
                self.stats[k].mul_(m).add_((1.0 - m) * s)
            else:
                self.stats[k].copy_(s)
                self.seen[k] = True

    # ------------------------------------------------------------------
    # encoder
    # ------------------------------------------------------------------
    def _encode(self):
        tok = self.token_proj(self.stats) + self.stage_emb.weight      # (K, d)
        h = self.out_norm(self.mixer(tok.unsqueeze(0))).squeeze(0)     # (K, d)
        logits = self.gap_head(torch.cat([h[:-1], h[1:]], dim=-1)).squeeze(-1)
        return h, logits + self.gap_offset                                               # (K,d), (K-1,)

    def _span_excess(self):
        return self.span_excess if self.fixed_span else F.softplus(self.span_raw)

    def forward(self):
        """Radii (K,), strictly increasing."""
        h, logits = self._encode()
        props = torch.softmax(logits, dim=0)
        gaps = self.min_gap + self._span_excess() * props
        k0 = self.floor + F.softplus(self.k0_head(h[0])).reshape(1)
        return torch.cat([k0, gaps]).cumsum(dim=0)

    @torch.no_grad()
    def gap_proportions(self):
        """Softmax proportions p_t — the share of the gap budget each adjacent
        stage pair receives (the quantity to report for 'relative distance')."""
        _, logits = self._encode()
        return torch.softmax(logits, dim=0)

    # ------------------------------------------------------------------
    # losses (same API as LearnableRadii)
    # ------------------------------------------------------------------
    def ceiling_loss(self, radii=None):
        if radii is None:
            radii = self.forward()
        return self.ceiling_weight * F.relu(radii[-1] - self.ceiling)

    @staticmethod
    def _trimmed_mean(vals, trim_ratio):
        n = vals.numel()
        if n == 0:
            return None
        k = int(n * trim_ratio)
        if 2 * k >= n:
            k = (n - 1) // 2
        if k <= 0:
            return vals.mean()
        sorted_vals, _ = torch.sort(vals)
        return sorted_vals[k:n - k].mean()

    def alignment_loss(self, features, labels, radii=None, c=None):
        """L_align = (1/K) Σ_c (ρ̄_c^{detach} − k_c)², ρ̄_c a trimmed mean.

        `c` is accepted for call-site compatibility and ignored — the
        curvature is already bound into norm_fn.
        """
        if radii is None:
            radii = self.forward()
        rho = self.norm_fn(features).detach()
        loss = torch.zeros((), device=radii.device)
        count = 0
        for k in range(self.num_classes):
            mask = labels == k
            if mask.sum() == 0:
                continue
            loss = loss + (self._trimmed_mean(rho[mask], self.trim_ratio) - radii[k]) ** 2
            count += 1
        return loss / max(count, 1)

    def extra_repr(self):
        with torch.no_grad():
            r = self.forward().cpu()
            p = self.gap_proportions().cpu()
        return (f"K={self.num_classes}, min_gap={self.min_gap}, floor={self.floor}, "
                f"span={'fixed' if self.fixed_span else 'learnable'}, "
                f"radii=[{', '.join(f'{x:.3f}' for x in r)}], "
                f"gap_props=[{', '.join(f'{x:.3f}' for x in p)}]")
