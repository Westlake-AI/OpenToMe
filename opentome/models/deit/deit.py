# /opentome/models/deit/deit.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_
from timm.models.registry import register_model


class DeiTModel(nn.Module):
    """
    Data-Efficient Image Transformer (DeiT) model.
    Based on VisionTransformer from timm, following mergenet/model.py style.
    """
    
    arch_zoo = {
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 384,
                         'depth': 12,
                         'num_heads': 6,
                         'mlp_ratio': 4.0
                        }),
        **dict.fromkeys(['s_ext', 'small_extend'],
                        {'embed_dims': 384,
                         'depth': 16,
                         'num_heads': 6,
                         'mlp_ratio': 4.0
                        }),
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 img_size=224,
                 patch_size=16,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 num_classes=1000,
                 pretrained=None):
        super().__init__()

        # arch setups
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            raise ValueError("Wrong setups.")
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = self.arch_settings['embed_dims']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratio = self.arch_settings['mlp_ratio']
        self.depth = self.arch_settings['depth']

        # Create VisionTransformer model (DeiT is based on ViT architecture)
        # Note: We don't pass **kwargs to avoid unsupported arguments like pretrained_cfg
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Load pretrained weights if provided
        if pretrained:
            self._load_pretrained_weights(pretrained, img_size, patch_size)

        self.num_classes = num_classes

    def _load_pretrained_weights(self, pretrained, img_size, patch_size):
        """Load pretrained weights from timm DeiT model."""
        import traceback
        from opentome.models.utils import load_pt_weights
        
        try:
            from timm.models import create_model
            
            # Determine model name based on architecture
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # Auto-determine model name
                if self.embed_dim == 384 and self.num_heads == 6:
                    model_name = 'deit_small_patch16_224'
                elif self.embed_dim == 768 and self.num_heads == 12:
                    model_name = 'deit_base_patch16_224'
                elif self.embed_dim == 192 and self.num_heads == 3:
                    model_name = 'deit_tiny_patch16_224'
                else:
                    print(f"Warning: Cannot auto-determine pretrained model. Specify model name explicitly.")
                    return
            
            print(f"Loading pretrained weights from: {model_name}")
            
            # Create pretrained model and extract weights
            pretrained_model = create_model(model_name, pretrained=True, img_size=img_size, num_classes=0)
            pretrained_state = pretrained_model.state_dict()
            
            # Load weights using utility function
            load_pt_weights(
                target_vit=self.vit,
                pretrained_state=pretrained_state,
                start_block=0,
                end_block=self.depth,
                verbose=True
            )
            
            print(f"Pretrained weights loaded successfully from {model_name}")
                    
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            print(f"Exception traceback:")
            traceback.print_exc()

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        return self.vit(x)


@register_model
def deit_s(pretrained=False, **kwargs):
    """DeiT Small model (12 blocks)"""
    # Filter out unsupported kwargs that timm might pass (like pretrained_cfg)
    supported_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in ['img_size', 'patch_size', 'drop_rate', 'attn_drop_rate', 
                 'drop_path_rate', 'num_classes']
    }
    model = DeiTModel(arch='small', pretrained=pretrained, **supported_kwargs)
    # 在断点前打印patch_size
    patch_size = supported_kwargs.get('patch_size', getattr(model, 'patch_size', None))
    print(f'[DEBUG] patch_size: {patch_size}')
    import pdb; pdb.set_trace()
    return model


@register_model
def deit_s_extend(pretrained=False, **kwargs):
    """DeiT Small Extended model (16 blocks)"""
    # Filter out unsupported kwargs that timm might pass (like pretrained_cfg)
    supported_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in ['img_size', 'patch_size', 'drop_rate', 'attn_drop_rate', 
                 'drop_path_rate', 'num_classes']
    }
    model = DeiTModel(arch='s_ext', pretrained=pretrained, **supported_kwargs)
    return model