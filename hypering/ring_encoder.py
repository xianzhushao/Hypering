"""Ring encoder: hyperbolic MLP encoder + ring geometry on the Poincare ball.

x (FC edge vector) -> LayerNorm -> HypLift (exp_0) -> 2 x HypMLPLayer -> HypLinear
-> embedding e on the Poincare ball D^d (c = 1).

Every stage c has a ring of hyperbolic radius k_c (k_0 < k_1 < ... < k_{K-1},
CN innermost, AD outermost). A sample embedded at e is scored by how close its
hyperbolic distance from the origin, rho(e) = d_D(0, e), is to each ring:

    a_c(e) = -(rho(e) - k_c)^2          affinity
    y_hat  = argmax_c a_c(e)            prediction = nearest ring
    L      = CE(a / tau, y)             ring loss
"""
import os

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Ring geometry
# ----------------------------------------------------------------------------
_BALLS = {}


def _ball(c=1.0, device=None):
    key = (float(c), str(device))
    ball = _BALLS.get(key)
    if ball is None:
        ball = geoopt.PoincareBall(c=float(c))
        if device is not None:
            ball = ball.to(device)
        _BALLS[key] = ball
    return ball


def log_map_zero(y, c=1.0):
    """Ball -> tangent space at the origin."""
    return _ball(c, y.device).logmap0(y)


def hyperbolic_norm(x, c=1.0):
    """rho(x) = d_D(0, x) = (2/sqrt(c)) * atanh(sqrt(c) * ||x||)."""
    return _ball(c, x.device).dist0(x)


def affinity_logits(features, radii, c=1.0):
    """(N, D) ball points, (K,) radii -> (N, K) affinities -(rho - k_c)^2."""
    rho = hyperbolic_norm(features, c)
    return -(rho.unsqueeze(1) - radii.unsqueeze(0)) ** 2


def affinity_predict(features, radii, c=1.0):
    """Nearest-ring prediction."""
    return torch.argmax(affinity_logits(features, radii, c=c), dim=-1)


def hyperbolic_ring_loss(features, radii, labels, c=1.0, tau=1.0):
    """Ring loss L = CE(a / tau, y)."""
    return F.cross_entropy(affinity_logits(features, radii, c=c) / tau, labels)


# ----------------------------------------------------------------------------
# Hyperbolic MLP encoder
# ----------------------------------------------------------------------------
class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.manifold = geoopt.PoincareBall(c=self.c)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = geoopt.ManifoldParameter(
                torch.zeros(out_features), manifold=self.manifold)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x)
        if self.bias is not None:
            mv = self.manifold.mobius_add(mv, self.bias)
        return self.manifold.projx(mv)


def _tangent_cap(v, max_norm):
    """Scale a tangent vector so its norm never exceeds max_norm (boundary-safe)."""
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return v * (n.clamp(max=max_norm) / n)


class HypLift(nn.Module):
    """Euclidean node features → Poincaré ball, with tangent-norm control.

    Treats each node's (dim-D) feature row as a tangent vector at the origin and
    maps it onto the ball with a single expmap0. The ÷√dim scaling plus a
    learnable gain stop expmap0's tanh from saturating (a node vector of D
    LayerNorm'd coords has norm ≈ √D, which without scaling would pin every node
    to the boundary), and the cap guarantees a strictly-interior lift.
    """
    def __init__(self, dim, c=1.0, max_tan_norm=2.0, init_gain=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.gain = nn.Parameter(torch.tensor(float(init_gain)))
        self.inv_sqrt = float(dim) ** -0.5
        self.max_tan_norm = max_tan_norm

    def forward(self, h):                                  # (B, N, dim) euclidean
        v = h * (self.inv_sqrt * self.gain)
        v = _tangent_cap(v, self.max_tan_norm)
        return self.manifold.projx(self.manifold.expmap0(v))


class HypMLPLayer(nn.Module):
    """One fully-hyperbolic mirror of `Linear → ReLU → BatchNorm1d → Dropout`.

    Same origin-tangent-space recipe HypGCNLayer uses, minus the neighbour
    aggregation (there is no graph here):

        u   = log0(h)                    # ball → T_0
        z   = u Wᵀ + b                   # the Linear
        z   = ReLU(z)                    # the activation, ReLU to match the baseline
        z   = BatchNorm1d(z)             # the BatchNorm, taken in T_0
        z   = dropout(z)                 # the Dropout
        out = projx(expmap0(cap(z)))     # T_0 → ball

    Order matters and is the baseline's, not the textbook one: the baseline
    puts BatchNorm *after* the activation, so this does too. Normalising in the
    tangent space rather than on the ball is what makes BatchNorm meaningful at
    all — the ball is not a vector space, so a running mean of ball points is
    not a ball point.
    """
    def __init__(self, in_dim, out_dim, c=1.0, dropout=0.3, max_tan_norm=2.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.max_tan_norm = max_tan_norm

    def forward(self, h):                                  # (B, in_dim) on ball
        z = self.lin(self.manifold.logmap0(h))
        z = F.relu(z)
        # BatchNorm1d needs ≥2 rows to estimate a batch statistic; a stray
        # single-sample batch would otherwise crash training rather than the
        # layer simply passing through.
        if not (self.training and z.shape[0] < 2):
            z = self.bn(z)
        z = self.drop(z)
        return self.manifold.projx(
            self.manifold.expmap0(_tangent_cap(z, self.max_tan_norm)))


class HypBaselineMLPEncoder(nn.Module):
    """Fully-hyperbolic counterpart of the Euclidean MLP baseline
    (Linear -> ReLU -> BN -> Dropout, twice, then Linear).

    Args:
        input_dim:    number of input features.
        hidden:       width of the first hidden layer; the second is hidden//2.
                      None reproduces the baseline's min(128, max(64, in//2)).
        out_dim:      Poincaré ball embedding dimension (= centers_dim).
        c:            curvature (fixed).
        dropout:      dropout rate inside both hidden layers (baseline: 0.3).
        max_tan_norm: tangent-norm cap enforced at every ball-valued stage.
    """
    def __init__(self, input_dim, hidden=None, out_dim=20, c=1.0, dropout=0.3,
                 max_tan_norm=2.0, init_gain=1.0):
        super().__init__()
        self.c = c
        h = int(hidden) if hidden else min(128, max(64, input_dim // 2))
        h2 = max(1, h // 2)
        self.hidden_dims = (h, h2)
        self.feat_norm = nn.LayerNorm(input_dim)
        self.lift = HypLift(input_dim, c=c, max_tan_norm=max_tan_norm,
                            init_gain=init_gain)
        self.layer1 = HypMLPLayer(input_dim, h, c=c, dropout=dropout,
                                  max_tan_norm=max_tan_norm)
        self.layer2 = HypMLPLayer(h, h2, c=c, dropout=dropout,
                                  max_tan_norm=max_tan_norm)
        # The baseline's bare final Linear: no activation, no norm, no dropout.
        self.out = HypLinear(h2, out_dim, c=c)

    def forward(self, x, isTest=False):
        if x.dim() == 1:                                   # single sample
            x = x.unsqueeze(0)
        h = self.lift(self.feat_norm(x))                   # euclidean → ball
        h = self.layer1(h)
        h = self.layer2(h)
        output = self.out(h)                               # ball → D^out_dim
        return output.squeeze().unsqueeze(0) if isTest else output.squeeze()


# ----------------------------------------------------------------------------
# Pretrained ring model
# ----------------------------------------------------------------------------
class RingModel(nn.Module):
    """Hyperbolic encoder f: R^10878 -> D^d plus K fixed (learned) ring radii."""

    def __init__(self, input_dim, centers_dim, radii, class_names,
                 c=1.0, hidden=None, dropout=0.3, max_tan_norm=2.0):
        super().__init__()
        self.c = c
        self.class_names = list(class_names)
        self.encoder = HypBaselineMLPEncoder(
            input_dim=input_dim, hidden=hidden, out_dim=centers_dim, c=c,
            dropout=dropout, max_tan_norm=max_tan_norm)
        self.register_buffer('radii', torch.as_tensor(radii, dtype=torch.float32))

    @torch.no_grad()
    def embed(self, x):
        """(N, input_dim) features -> (N, d) points on the Poincare ball."""
        e = self.encoder(x)
        return e.unsqueeze(0) if e.dim() == 1 else e

    @torch.no_grad()
    def forward(self, x):
        """Returns a dict with embedding, rho = d(0, e), affinities and prediction."""
        e = self.embed(x)
        a = affinity_logits(e, self.radii, c=self.c)
        return {'embedding': e,
                'rho': hyperbolic_norm(e, self.c),
                'affinity': a,
                'pred': a.argmax(dim=-1)}


def load_pretrained(run_dir, device='cpu'):
    """Load one trained run, e.g. checkpoints/seed_042."""
    ckpt = torch.load(os.path.join(run_dir, 'model.pt'), map_location=device)
    cfg = ckpt['config']
    model = RingModel(input_dim=cfg['input_dim'], centers_dim=cfg['centers_dim'],
                      radii=ckpt['radii'], class_names=ckpt['class_names'],
                      c=cfg['curvature'], hidden=cfg.get('hidden'),
                      dropout=cfg['dropout'], max_tan_norm=cfg['max_tan_norm'])
    model.encoder.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()
