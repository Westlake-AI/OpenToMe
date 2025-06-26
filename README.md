# Toolbox and Benchmark for Token Merging Modules

## News

- **2025-06-23**: Setup the basic framework of OpenToMe.

## Installation

### Install from source
```bash
git git@github.com:Westlake-AI/OpenToMe.git
cd OpenToMe
pip install -e .
```

### Install experiment dependencies

```bash
pip install -r requirements.txt
```

## Getting Started

### Model Examples

Here is an example of using ToMe with timm Attention blocks.

```python
import torch
import timm
from torch import nn
from opentome.timm import Block, tome_apply_patch
from opentome.tome import check_parse_r


class TransformerBlock(nn.Module):
    def __init__(self, *, embed_dim=768, num_layers=12, num_heads=12, drop_path=0.0,
                 with_cls_token=True, init_values=1e-5, use_flash_attn=False, **kwargs):
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token

        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        dp_rates=[x.item() for x in torch.linspace(drop_path, 0.0, num_layers)]
        self.blocks = nn.Sequential(
            *[Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                init_values=init_values,
                drop_path=dp_rates[j],
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                mlp_layer=timm.layers.Mlp,
                use_flash_attn=use_flash_attn,
            ) for j in range(num_layers)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

# test
embed_dim, token_num, merge_num, inflect = 384, 196, 100, 0.5
x = torch.randn(1, token_num, embed_dim)
# model = timm.create_model('vit_small_patch16_224')
model = TransformerBlock(embed_dim=384, num_layers=12, num_heads=8)
z = model.forward(x)
print(x.shape, z.shape)

# update tome
merge_ratio = check_parse_r(len(model.blocks), merge_num, token_num, inflect)
tome_apply_patch(model)
model.r = (merge_ratio, inflect)
model._tome_info["r"] = model.r
model._tome_info["total_merge"] = merge_num

z = model.forward(x)
print(x.shape, z.shape)
```

### ImageNet Image Classification

Here is an example of evaluate ImageNet validation set with various Token Compression methods.
```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
./evaluations/image_classification/in1k_example.py --merge_num 100 --dataset ./data/ImageNet/val
```

## Token Compression Baselines

- [x] **ToMe [ICLR 2023]** Token Merging: Your ViT but Faster [paper](https://arxiv.org/abs/2210.09461) [code](https://github.com/facebookresearch/ToMe)
- [x] **DiffRate [ICCV2023]** Diffrate: Differentiable Compression Rate for Efficient Vision Transformers [paper](https://arxiv.org/abs/2305.17997) [code](https://github.com/OpenGVLab/DiffRate)
- [x] **DTEM [NIPS2024]** Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers [paper](https://openreview.net/forum?id=pVPyCgXv57) [code](https://github.com/movinghoon/DTEM)
- [x] **ToFu [WACV2024]** Token Fusion: Bridging the Gap between Token Pruning and Token Merging [paper](https://arxiv.org/abs/2312.01026)
- [x] **MCTF [CVPR2024]** Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers [paper](https://arxiv.org/abs/2403.10030) [code](https://github.com/mlvlab/MCTF)
- [ ] **CrossGET [ICML2024]** CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers [paper](https://arxiv.org/abs/2305.17455) [code](https://github.com/sdc17/CrossGET)
- [ ] **PiToMe [NIPS2024]** Accelerating Transformers with Spectrum-Preserving Token Merging [paper](https://arxiv.org/abs/2405.16148) [code](https://github.com/hchautran/PiToMe)
- [ ] **DCT [ACL2023]** Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator [paper](https://arxiv.org/abs/2305.15099) [code](https://github.com/LUMIA-Group/FourierTransformer)


<p align="right">(<a href="#top">back to top</a>)</p>
