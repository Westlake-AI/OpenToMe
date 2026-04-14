"""MuonC1b: Muon with Nesterov correction applied AFTER Newton-Schulz (orthogonal space).

Pipeline comparison:
  A6:  buf(raw) → Nesterov(raw) → NS → A6-scale → update
  C1b: buf(raw) → NS(buf)       → Nesterov(ortho, finite-diff) → A6-scale → update

Key insight: In A6, Nesterov mixes noisy raw gradients before orthogonalization.
In C1b, NS first produces a clean orthogonal direction; Nesterov then extrapolates
consecutive NS directions in that clean space — more principled lookahead.

Finite-difference Nesterov in orthogonal space:
  u_final = u + beta * (u - prev_u)

Extra memory: one float32 buffer per Muon parameter (prev_u).
NS calls: 1 per step (same as A6).
LR scaling: A6-style (max(1, rows/cols)^0.5), unchanged from base Muon.

SAC ablation result: PPL 16.168 vs A6=16.208 (+0.040 improvement).
Recommended lr: 6e-3.
"""

import torch
from ..muon import Muon, zeropower_via_newtonschulz5


class MuonC1b(Muon):
    """C1b: Nesterov correction applied AFTER Newton-Schulz orthogonalization.

    SAC Part C ablation winner; improvement of +0.040 PPL over A6 baseline.
    Orthogonal to S4 (scaling vs timing), effects are approximately additive.
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

                # Step M: EMA on raw gradient (same as A6)
                buf.lerp_(g, 1 - momentum)

                # Step O: NS on momentum buffer (not Nesterov-corrected gradient)
                u = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])

                # Step N_ortho: Nesterov finite-difference in orthogonal space
                u_f = u.float()
                if "prev_u" not in state:
                    state["prev_u"] = torch.zeros_like(u_f)
                prev_u = state["prev_u"]
                update = u_f + momentum * (u_f - prev_u)
                state["prev_u"].copy_(u_f)

                # Step S: A6-style LR scaling (unchanged from base Muon)
                if update.ndim >= 2:
                    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5

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
