# Toolbox and Benchmark for Token Merging Modules

## News

- **2025-06-23**: Setup the basic framework of OpenToMe.

## Installation

### Install experimental dependencies

```bash
conda create -n opentome python=3.10.0
conda activate opentome
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Install OpenToMe from source
```bash
git git@github.com:Westlake-AI/OpenToMe.git
cd OpenToMe
pip install -e .
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

### Image Classification on ImageNet

Here is an example of evaluate ImageNet validation set with various Token Compression methods.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
./evaluations/image_classification/in1k_example.py \
--model_name vit_base_patch16_224 \
--tome tome \
--merge_num 100 \
--dataset /PATH/TO/ImageNet/val \
--inflect -0.5 \
```
You can also run the evaluation with the bash example on GPU0:
```bash
bash evaluations/image_classification/in1k_eval.sh 0 tome 100 /PATH/TO/ImageNet/val 1 deit_small_patch16_224
```

### Image ToMe Visualization
Here is an example of visualization with various Token Compression methods
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=0 \
evaluations/visualizations/vis_classification.py \
--image_path ./demo \
--model_name deit_small_patch16_224 \
--tome tome \
--merge_num 100 \
--inflect -0.5 \
--save_vis True \
```
You can run the visualization with the bash example:
```bash
bash evaluations/visualizations/vis_eval.sh ./demo tome 100 1 deit_small_patch16_224
```

## Token Compression Baselines

- [x] **ToMe [ICLR'23]** Token Merging: Your ViT but Faster [paper](https://arxiv.org/abs/2210.09461) [code](https://github.com/facebookresearch/ToMe)
- [x] **DiffRate [ICCV'23]** Diffrate: Differentiable Compression Rate for Efficient Vision Transformers [paper](https://arxiv.org/abs/2305.17997) [code](https://github.com/OpenGVLab/DiffRate)
- [x] **DCT [ACL'23]** Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator [paper](https://arxiv.org/abs/2305.15099) [code](https://github.com/LUMIA-Group/FourierTransformer)
- [ ] **LTMP [TMLR'23]** Learned Thresholds Token Merging and Pruning for Vision Transformers [paper](https://arxiv.org/pdf/2307.10780) [code](https://github.com/Mxbonn/ltmp)
- [x] **ToFu [WACV'24]** Token Fusion: Bridging the Gap between Token Pruning and Token Merging [paper](https://arxiv.org/abs/2312.01026)
- [ ] **CrossGET [ICML'24]** CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers [paper](https://arxiv.org/abs/2305.17455) [code](https://github.com/sdc17/CrossGET)
- [x] **MCTF [CVPR'24]** Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers [paper](https://arxiv.org/abs/2403.10030) [code](https://github.com/mlvlab/MCTF)
- [x] **ATC [ECCV'24]** [paper](https://arxiv.org/abs/2409.11923) [code](https://github.com/JoakimHaurum/ATC)
- [x] **DTEM [NeurIPS'24]** Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers [paper](https://openreview.net/forum?id=pVPyCgXv57) [code](https://github.com/movinghoon/DTEM)
- [x] **PiToMe [NeurIPS'24]** Accelerating Transformers with Spectrum-Preserving Token Merging [paper](https://arxiv.org/abs/2405.16148) [code](https://github.com/hchautran/PiToMe)
- [x] **FPET [CVPR'25]** Faster Parameter-Efficient Tuning with Token Redundancy Reduction [paper](https://arxiv.org/abs/2503.20282) [code](https://github.com/kyk120/fpet)

## Support Tasks (TODO List)

- [x] Image Classification
- [ ] Image Generation
- [ ] M/LLM Inference 
- [ ] Long Sequence
- [x] Throughput
- [ ] AI for Science
- [x] ToMe Visualization

## Summary Functions

| Method | Published | Support Source Matrix/Map | Support Visualization|
|:------:|:---------:|:-------------------------:|:--------------------:|
| ToMe   | ICLR’23   | Matrix, Map | ✅ |
| DiffRate | ICCV’23 | Matrix      | ✅ |
| DCT | ACL’23 | None | ❌ |
| ToFu | WACV’24 | Matrix, Map | ✅ |
| MCTF | CVPR’24 | Matrix, Map | ✅ |
| ATC | ECCV’24 | Map | ❌ |
| DTEM | NIPS’24 | Matrix, Map | ✅ |
| PiToMe | NIPS’24 | Matrix | ✅ |
| FPET | CVPR’25 | Matrix | ✅ |

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation:

```bib
@article{2025opentome,
  title = {OpenToMe},
  author = {Siyuan Li and Xin Jin and Kai Yu},
  year = {2025},
  url={https://github.com/Westlake-AI/OpenToMe},
  urldate = {2025-08-15},
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
