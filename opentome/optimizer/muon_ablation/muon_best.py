"""MuonBest: Combines S4 symmetric scaling + C1b post-NS Nesterov.

S4 and C1b are orthogonal changes (scaling formula vs Nesterov timing),
so their benefits are approximately additive.

Full pipeline:
  Step M:  buf = lerp(buf, g, 1-beta)            # EMA in raw gradient space
  Step O:  u = NS(buf, steps=5)                  # NS on momentum buffer (C1b)
  Step N:  u_final = u + beta*(u - prev_u)        # Nesterov in ortho space (C1b)
           prev_u <- u
  Step S:  update = u_final * (max(r,c)/min(r,c))^0.5   # S4 symmetric scaling
  Step W:  p = p * (1 - lr * wd)                 # decoupled weight decay
  Step U:  p = p - lr * update                   # parameter update

SAC reference: MuonBestS4C1b, lr=6e-3.
Expected PPL improvement over A6: ~0.12 (0.080 from S4 + 0.040 from C1b).
Recommended lr: 6e-3.
"""

import torch
from ..muon import Muon, zeropower_via_newtonschulz5
from .scaling import _scale_s4


class MuonBest(Muon):
    """MuonBest: S4 symmetric scaling + C1b post-NS Nesterov combined.

    Current best candidate from SAC ablation study.
    Improvements are approximately additive since changes are orthogonal.
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

                # Step M: EMA on raw gradient
                buf.lerp_(g, 1 - momentum)

                # Step O: NS on momentum buffer (C1b: NS before Nesterov)
                u = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])

                # Step N_ortho: Nesterov in orthogonal space (C1b)
                u_f = u.float()
                if "prev_u" not in state:
                    state["prev_u"] = torch.zeros_like(u_f)
                prev_u = state["prev_u"]
                update = u_f + momentum * (u_f - prev_u)
                state["prev_u"].copy_(u_f)

                # Step S: S4 symmetric scaling
                if update.ndim >= 2:
                    update = update * _scale_s4(update)

                # Step W + U
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
