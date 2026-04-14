"""MuonS4: Muon with symmetric aspect-ratio LR scaling.

Scaling formula: scale = (max(r,c) / min(r,c))^0.5  (applied in-place on update)

Key difference from A6 (base Muon):
  A6:  scale = max(1, rows/cols)^0.5  — wide matrices get scale=1.0 (no boost)
  S4:  scale = (max/min)^0.5          — wide matrices get same boost as tall matrices

Affected case — FFN up-proj [h, 4h]:
  A6 scale=1.0,  S4 scale=2.0

SAC ablation result: PPL 16.128 vs A6=16.208 (+0.080 improvement).
Recommended lr: 6e-3.
"""

import torch
from ..muon import Muon, zeropower_via_newtonschulz5
from .scaling import _scale_s4


class MuonS4(Muon):
    """S4: Symmetric aspect-ratio LR scaling — (max(r,c)/min(r,c))^0.5.

    Both tall and wide matrices are scaled by the aspect ratio.
    SAC Part B ablation winner; improvement of +0.080 PPL over A6 baseline.
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

                # S4: symmetric scaling
                if update.ndim >= 2:
                    update = update * _scale_s4(update)

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update.reshape(p.shape), alpha=-lr)

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
