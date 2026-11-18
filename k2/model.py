"""Encoder, predictor and decoder (PyTorch).

Encoder: a frame tokenizer (63 tile tokens, 12 item tokens, 4 vital tokens, facing, light, sleeping), a transformer,
cross-attention pooling with Q learned queries (Q x query_dim) and a linear projector into the latent. No final
LayerNorm: SIGReg could not shape the latent through one.

Predictor: a causal transformer over the C context steps of (z, m); residual output z_{j+1} = z_j + f(...).

Decoder: one learned slot per token cross-attends to the latent (split into a few memory tokens); per-token heads give
block logits and mob logits per tile, count logits per item, value logits per vital, facing logits, light, sleeping.
"""

from __future__ import annotations

import torch
from torch import nn

from . import layout as L


def _transformer(width, layers, heads):
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Tokenizer(nn.Module):
    """Frames (leading dims ...) -> tokens (..., 82, width)."""

    def __init__(self, width: int):
        super().__init__()
        p = lambda *s: nn.Parameter(torch.randn(*s) * 0.02)  # noqa: E731
        self.block, self.mob, self.tile_pos = nn.Embedding(L.NUM_BLOCKS, width), p(L.NUM_MOBS, width), p(L.TILES, width)
        self.item, self.count = p(L.NUM_ITEMS, width), nn.Embedding(L.LEVELS, width)
        self.vital, self.value = p(L.NUM_VITALS, width), nn.Embedding(L.LEVELS, width)
        self.facing, self.facing_type = nn.Embedding(4, width), p(width)
        self.light_type, self.light = p(width), nn.Linear(1, width)
        self.sleep, self.sleep_type = nn.Embedding(2, width), p(width)

    def forward(self, f: dict) -> torch.Tensor:
        tiles = self.block(f["block"]) + f["mob"] @ self.mob + self.tile_pos
        items = self.item + self.count(f["item"])
        vitals = self.vital + self.value(f["vital"])
        facing = (self.facing(f["facing"]) + self.facing_type)[..., None, :]
        light = (self.light(f["light"][..., None]) + self.light_type)[..., None, :]
        sleep = (self.sleep(f["sleep"]) + self.sleep_type)[..., None, :]
        return torch.cat([tiles, items, vitals, facing, light, sleep], -2)


class Encoder(nn.Module):
    """Frames (...) -> latents (..., latent)."""

    def __init__(self, width=128, layers=3, heads=4, queries=4, query_dim=64, latent=256):
        super().__init__()
        self.tokenizer = Tokenizer(width)
        self.body = _transformer(width, layers, heads)
        self.norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.down = nn.Linear(width, query_dim)
        self.project = nn.Linear(queries * query_dim, latent)
        self.latent = latent

    def forward(self, f: dict) -> torch.Tensor:
        tokens = self.tokenizer(f)
        lead = tokens.shape[:-2]
        x = self.norm(self.body(tokens.reshape(-1, *tokens.shape[-2:])))
        q = self.queries.expand(len(x), -1, -1)
        pooled = self.down(self.pool(q, x, x, need_weights=False)[0])
        return self.project(pooled.flatten(1)).reshape(*lead, self.latent)


class Predictor(nn.Module):
    """(z_0 .. z_{C-1}, m_0 .. m_{C-1}) -> predictions of z_1 .. z_C, each from its causal past."""

    def __init__(self, latent=256, k=1, context=1, width=256, layers=3, heads=4):
        super().__init__()
        self.k, self.context = k, context
        self.z_in = nn.Linear(latent, width)
        self.m_in = nn.Sequential(nn.Linear(k * L.NUM_ACTIONS, width), nn.GELU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.randn(context, width) * 0.02)
        self.body = _transformer(width, layers, heads)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, latent)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """z (B, c, D), m (B, c, k) action indices, c <= C -> (B, c, D)."""
        c = z.shape[1]
        m = nn.functional.one_hot(m, L.NUM_ACTIONS).float().flatten(-2)
        x = self.z_in(z) + self.m_in(m) + self.pos[:c]
        mask = nn.Transformer.generate_square_subsequent_mask(c, device=z.device)
        h = self.norm(self.body(x, mask=mask, is_causal=True))
        return z + self.out(h)

    def rollout(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Context latents z (B, c, D) and the macro-actions m (B, c - 1 + H, k) from the first context step on ->
        the predicted latents of the H steps after the context (B, H, D), each fed back."""
        zs = list(z.unbind(1))
        c = len(zs)
        for h in range(m.shape[1] - c + 1):
            ctx = torch.stack(zs[-self.context :], 1)
            end = c + h
            zs.append(self(ctx, m[:, end - ctx.shape[1] : end])[:, -1])
        return torch.stack(zs[c:], 1)


HEADS = {  # name: (tokens, logits per token)
    "tile": (L.TILES, L.NUM_BLOCKS + L.NUM_MOBS),
    "item": (L.NUM_ITEMS, L.LEVELS),
    "vital": (L.NUM_VITALS, L.LEVELS),
    "facing": (1, 4),
    "light": (1, 1),
    "sleep": (1, 1),
}
OUT_DIM = sum(t * d for t, d in HEADS.values())


def split_heads(x: torch.Tensor) -> dict:
    """A flat output (..., OUT_DIM) -> the decoder's dict: block (..., 63, 17), mob (..., 63, 4), item (..., 12, 10),
    vital (..., 4, 10), facing (..., 4), light (...), sleep (...) (logits; light is a value)."""
    out, o = {}, 0
    for name, (t, d) in HEADS.items():
        out[name] = x[..., o : o + t * d].unflatten(-1, (t, d))
        o += t * d
    return {
        "block": out["tile"][..., : L.NUM_BLOCKS],
        "mob": out["tile"][..., L.NUM_BLOCKS :],
        "item": out["item"],
        "vital": out["vital"],
        "facing": out["facing"][..., 0, :],
        "light": out["light"][..., 0, 0],
        "sleep": out["sleep"][..., 0, 0],
    }


class Decoder(nn.Module):
    """Latents (..., D) -> per-token outputs (see `split_heads`)."""

    def __init__(self, latent=256, width=128, layers=2, heads=4, memory=4):
        super().__init__()
        self.memory, self.width = memory, width
        self.to_memory = nn.Linear(latent, memory * width)
        self.slots = nn.Parameter(torch.randn(L.NUM_TOKENS, width) * 0.02)
        layer = nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleDict({n: nn.Linear(width, d) for n, (_, d) in HEADS.items()})
        self.ranges = {"tile": L.TILE_TOKENS, "item": L.ITEM_TOKENS, "vital": L.VITAL_TOKENS,
                       "facing": slice(L.FACING_TOKEN, L.FACING_TOKEN + 1),
                       "light": slice(L.LIGHT_TOKEN, L.LIGHT_TOKEN + 1),
                       "sleep": slice(L.SLEEP_TOKEN, L.SLEEP_TOKEN + 1)}  # fmt: skip

    def forward(self, z: torch.Tensor) -> dict:
        lead = z.shape[:-1]
        mem = self.to_memory(z.reshape(-1, z.shape[-1])).unflatten(-1, (self.memory, self.width))
        h = self.norm(self.body(self.slots.expand(len(mem), -1, -1), mem))
        flat = torch.cat([self.heads[n](h[:, r]).flatten(1) for n, r in self.ranges.items()], 1)
        return split_heads(flat.reshape(*lead, OUT_DIM))


class LinearDecoder(nn.Module):
    """The linear probe: one linear map from the latent to every token's outputs."""

    def __init__(self, latent=256):
        super().__init__()
        self.linear = nn.Linear(latent, OUT_DIM)

    def forward(self, z: torch.Tensor) -> dict:
        return split_heads(self.linear(z))


class World(nn.Module):
    """The modules of one variant: the encoder always, the predictor unless E0, the decoder in E0."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.width, cfg.layers, cfg.heads, cfg.queries, cfg.query_dim, cfg.latent)
        self.predictor = (
            Predictor(cfg.latent, cfg.k, cfg.context, cfg.pred_width, cfg.pred_layers, cfg.heads)
            if cfg.variant != "E0" else None
        )  # fmt: skip
        self.decoder = Decoder(cfg.latent, cfg.width, cfg.dec_layers, cfg.heads) if cfg.variant == "E0" else None
