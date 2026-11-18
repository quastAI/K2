"""The model side's data: a cache table on the device, and batches of level-2 steps in `frame` mode.

A level-2 step at stride k: boundary frames t_j = t_0 + j k, the macro-action m_j = (a_{t_j} .. a_{t_j + k - 1}).
A batch holds C + 1 boundary frames x_0 .. x_C and the C macro-actions between them, all inside one episode.
"""

from __future__ import annotations

import numpy as np
import torch

from . import layout
from .cache import starts


class Frames:
    """The compact frames and actions of one split, as tensors on `device`."""

    def __init__(self, table: dict, device="cpu"):
        self.table = table
        self.device = torch.device(device)
        self.t = {f: torch.from_numpy(np.ascontiguousarray(table[f])).to(self.device) for f in layout.FIELDS}
        self.action = torch.from_numpy(table["action"].astype(np.int64)).to(self.device)
        self.mob_bits = (1 << torch.arange(layout.NUM_MOBS, device=self.device)).to(torch.uint8)

    def __len__(self):
        return len(self.table["action"])

    def frames(self, rows: torch.Tensor) -> dict:
        """The frames at `rows` (any shape): integer fields as long, mob bits (..., 63, 4) and light as float."""
        rows = rows.to(self.device)
        t = self.t
        return {
            "block": t["block"][rows].long(),
            "mob": ((t["mob"][rows].unsqueeze(-1) & self.mob_bits) > 0).float(),
            "item": t["item"][rows].long(),
            "vital": t["vital"][rows].long(),
            "facing": t["facing"][rows].long(),
            "light": t["light"][rows].float(),
            "sleep": t["sleep"][rows].long(),
        }

    def starts(self, k: int, context: int, sources=None, extra: int = 0) -> torch.Tensor:
        """Frames t_0 from which C + extra level-2 steps of stride k stay inside the episode."""
        return torch.from_numpy(starts(self.table, k * (context + extra), sources)).to(self.device)

    def steps(self, t0: torch.Tensor, k: int, n: int) -> tuple[dict, torch.Tensor]:
        """From starts t0 (B,): the frames x_0 .. x_n (B, n + 1, ...) and the macro-actions m_0 .. m_{n-1} (B, n, k)."""
        t0 = t0.to(self.device)
        rows = t0[:, None] + k * torch.arange(n + 1, device=self.device)
        acts = self.action[t0[:, None, None] + k * torch.arange(n, device=self.device)[:, None]
                           + torch.arange(k, device=self.device)]  # fmt: skip
        return self.frames(rows), acts


def flat_state(f: dict) -> torch.Tensor:
    """The `state` vector (..., 1345) of torch frames (for the Hamming cost and checks)."""
    blocks = torch.nn.functional.one_hot(f["block"], layout.NUM_BLOCKS).float()
    view = torch.cat([blocks, f["mob"]], -1).flatten(-2)
    levels = torch.as_tensor(layout.LEVEL_VALUES, device=view.device)
    return torch.cat(
        [
            view,
            levels[f["item"]],
            levels[f["vital"]],
            torch.nn.functional.one_hot(f["facing"], 4).float(),
            f["light"][..., None],
            f["sleep"][..., None].float(),
        ],
        -1,
    )


def from_states(states: np.ndarray, device="cpu") -> dict:
    """Torch frames from raw observations (n, 1345), e.g. the simulator's."""
    c = layout.compress(np.asarray(states).reshape(-1, layout.STATE_DIM))
    tmp = Frames({**c, "action": np.zeros(len(c["block"]), np.int8)}, device)
    return tmp.frames(torch.arange(len(c["block"]), device=tmp.device))
