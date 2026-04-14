import torch
import math

class SCALE(torch.optim.Optimizer):
    """
     SCALE: Stochastic Column-normalized Last-layer Momentum Optimizer
    """

    def __init__(
        self,
        params,   # dummy parameter for compatibility with torch.optim.Optimizer
        lr=1e-4,
        weight_decay=0.0,
        main_params=None,
        secondary_params=None,
        oned_params=None,
        id_to_name=None,
        debug=False,
        momentum=0.9,
        adam_lr=None,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8, 
    ):
        if adam_lr is None:
            adam_lr = lr 
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            adam_lr=adam_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params = list(main_params)
        secondary_params = list(secondary_params) if secondary_params is not None else []
        params.extend(secondary_params)
        oned_params = list(oned_params) if oned_params is not None else []
        params.extend(oned_params)
        self.id_to_name = id_to_name
        self.debug = debug
        self.max_lr = lr
        super().__init__(params, defaults)

        # Use integer codes for param_type so that optimizer state_dict
        # is compatible with distributed checkpoint flattening
        # (which only supports tensor/int/float in optimizer state).
        MAIN_PARAM = 0
        SECONDARY_PARAM = 1
        ONED_PARAM = 2

        for p in main_params:
            self.state[p]["param_type"] = MAIN_PARAM
        for p in secondary_params:
            self.state[p]["param_type"] = SECONDARY_PARAM
        for p in oned_params:
            self.state[p]["param_type"] = ONED_PARAM

    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            
            #############################
            #   Main+Secondary Params   #
            #############################
            params = [
                p
                for p in group["params"]
                if self.state[p]["param_type"] in (0, 1)
            ]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["momentum"]

            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                orig_shape = p.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
 
                if self.debug:
                    print("Main: ", self.id_to_name[id(p)])
                
                if self.id_to_name is not None and "embed_tokens" in self.id_to_name[id(p)]:  
                    col_dim = 0
                else:
                    col_dim = 1
                
                # calc update
                state = self.state[p]

                if self.state[p]["param_type"] == 1:
                    # add momentum 
                    if "moment1" not in state:
                        state["moment1"] = torch.zeros_like(g)
                    buf1 = state["moment1"]
                    buf1.lerp_(g, 1 - beta1)
                    g = buf1

                var = torch.mean(torch.square(g), dim=col_dim, keepdim=True)
                s = torch.sqrt(var).clamp_min_(1e-8)
                u = g / s
                if u.shape != orig_shape:
                    u = u.view(orig_shape)

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)
                
                # apply update
                p.data.add_(u, alpha=-lr)
                
                if self.debug:
                    print("p.data.dtype: ", p.data.dtype, "u.dtype: ",  u.dtype)

            ############################
            #       Oned Params        #
            ############################
            params = [
                p
                for p in group["params"]
                if self.state[p]["param_type"] == 2
            ]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]
            
            lr =  group['lr']
            adam_lr = group['adam_lr'] 
            max_lr = self.max_lr
            lr = (adam_lr / max_lr) * lr 
            
            for p in params:          
                g = p.grad
                if g is None:
                    continue
                
                if self.debug:
                    print("1D (AdamW): ", self.id_to_name[id(p)])
                    print("Adam lr = ", lr) 
                
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)                

        return loss
