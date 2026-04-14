"""MuonOldScale: Muon with old absolute-size LR scaling.

Scaling formula: effective_lr = lr * 0.2 * sqrt(max(rows, cols))
Source: OldSAC/opt/muon.py (adjust_lr_for_muon)

Recommended lr: 1e-3
  For square [1024,1024]: effective = 1e-3 * 6.4 = 6.4e-3 ≈ new-style lr=6e-3.

All non-scaling operations use new-style implementation:
  - lerp_-based momentum (not mul_.add_)
  - @torch.no_grad() on step()
  - No @torch.compile (FSDP2 DTensor compatibility)
  - Supports G.ndim >= 2 (batched NS)
"""

import torch
from ..muon import Muon, zeropower_via_newtonschulz5
from .scaling import _scale_old


class MuonOldScale(Muon):
    """Muon with old absolute-size LR scaling: effective_lr = lr * 0.2 * sqrt(max(r, c)).

    Isolates the effect of the scaling formula: all other ops use new-style Muon.
    Weight decay uses base lr (not scaled), preserving original OldSAC behavior.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################
            params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf.lerp_(g, 1 - momentum)
                update = g.lerp_(buf, momentum) if group["nesterov"] else buf
                if update.ndim > 2:
                    update = update.view(len(update), -1)
                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])

                # OldScale: effective_lr = lr * 0.2 * sqrt(max(r, c))
                if update.ndim >= 2:
                    adjusted_lr = lr * _scale_old(update)
                else:
                    adjusted_lr = lr

                p.data.mul_(1 - lr * weight_decay)           # WD uses base lr
                p.data.add_(update.reshape(p.shape), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if not self.state[p].get("use_muon", False)]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                buf1, buf2 = state["moment1"], state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                scale = bc1 / bc2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
