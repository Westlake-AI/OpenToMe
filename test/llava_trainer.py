import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from torchvision import utils
import numpy as np

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional, Tuple

# TODO
# Please Make Sure you Import the Related Files
# Optimizer Baseline
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import SGD
from torch.optim import AdamW
from llava.model.optim.lamb_sgg import LambSGG
from llava.model.optim.lamb import Lamb
from llava.model.optim.muon import Muon
from llava.model.optim.shampoo import Shampoo
from llava.model.optim.soap import SOAP
from llava.model.optim.mars import MARS
from llava.model.optim.radam import RAdam
from llava.model.optim.nadam import NAdam
from llava.model.optim.adan import Adan
from llava.model.optim.sophia import SophiaG
from llava.model.optim.lion import Lion
from llava.model.optim.adam_mini import Adam_mini
from llava.model.optim.came import CAME
from llava.model.optim.galore_adamw import GaLore_AdamW
from llava.model.optim.conda import Conda
from llava.model.optim.apollo import APOLLO_AdamW

# Optimizer with SGG
from llava.model.optim.adamw_sgg import AdamWSGG
from llava.model.optim.adamw_v2 import AdamWSGG_v2
from llava.model.optim.adafactor_sgg import AdafactorSGG
from llava.model.optim.adafactor_v2 import AdafactorSGG_v2
from llava.model.optim.lamb_v2 import LambSGG_v2
from llava.model.optim.shampoo_v2 import ShampooSGG_v2

# # Optimizer with SAC
from llava.model.optim.adamw_sac import AdamWSAC
from llava.model.optim.shampoo_sac import ShampooSAC
from llava.model.optim.adam_mini_sac import Adam_miniSAC


from llava.model.augment.augmenter import AugmentedImageProcess

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    ###############
    # Ours....
    ###############
    def __init__(self, refer_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_augment = self.args.augment
        self.ranking_loss = self.args.ranking
        self.tome_merge_num = self.args.merge_num
        self.aug_type = self.args.augment_type
        self.loss_type = self.args.loss_type
        assert self.loss_type in ["simpo", "dpo"], "Unsupportable loss function."
        self.refer_model = refer_model
        # self.last_reference_step = 0
        # self.reference_update_interval = getattr(self.args, "reference_update_steps", 500)

        # --- Augmentation --- #
        self.base_vision_tok: int = 576
        self.use_attn = False
        if self.aug_type in ["mergemix", "mergemix-r"]:
            self.use_attn = True
        self.augmenter = AugmentedImageProcess(aug_type=self.aug_type, 
                                               merge_num=self.args.merge_num,
                                               use_attn=self.use_attn
                                            )

        print("**********LLaVA Trainer Statue**********") 
        print("Augmentation: {}\nAugmentation Type: {}\nAttention: {}\nRanking loss: {}\nLoss type: {}"\
               .format(self.use_augment, self.aug_type, self.use_attn, self.ranking_loss, self.loss_type))
    
    def get_scheduler(self, metric, iter):
        # TODO
        pass

    def lambda_scale(self, _mean, _std, _tao=1e-5):
        _lam = np.random.normal(_mean, _std)
        _lam = (_lam - np.min(_lam)) / (np.max(_lam) - np.min(_lam) + _tao)
        if np.any(_lam < 0):
            raise ValueError("Lambda values should be >= 0.")
        if np.any(_lam > 1):
            _lam = np.clip(_lam, 0, 1)
            print_log("Warning: Lambda values were clipped to [0,1] due to floating point precision issues.", logger='root')
        return _lam


    def get_batch_logps(self, 
                        logits: torch.FloatTensor,
                        labels: torch.LongTensor, 
                        return_per_token_logp=False, 
                        return_all=False) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        # Adding a small constant to logits for numerical stability
        logits = logits + 1e-9

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(2)).squeeze(2)

        log_prob = (per_token_logps * loss_mask).sum(-1)
        loss_mask_sum = loss_mask.sum(-1)

        return log_prob


    def simpo_loss(self, 
                   policy_chosen_logps: torch.FloatTensor,
                   policy_rejected_logps: torch.FloatTensor,
                   lengths_chosen: torch.FloatTensor,
                   lengths_rejected: torch.FloatTensor,
                   beta: float,
                   gamma: float) -> torch.FloatTensor:
        """Compute the SimPO loss for a batch of policy log probabilities and sequence lengths.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            lengths_chosen: Lengths of the chosen responses. Shape: (batch_size,)
            lengths_rejected: Lengths of the rejected responses. Shape: (batch_size,)
            beta: Scaling parameter for the SimPO loss.
            gamma: Margin parameter for the SimPO loss.

        Returns:
            A tensor containing the SimPO loss for each example in the batch.
        """
        normalized_chosen_logps = policy_chosen_logps / lengths_chosen
        normalized_rejected_logps = policy_rejected_logps / lengths_rejected
        if torch.isnan(normalized_chosen_logps).any():
            print(policy_chosen_logps)
        if torch.isnan(normalized_rejected_logps).any():
            print(policy_rejected_logps)
        logits = (normalized_chosen_logps - normalized_rejected_logps) - gamma
        losses = -F.logsigmoid(beta * logits).mean()
        # losses = -F.logsigmoid(logits).mean()

        return losses
    
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        beta: float = 1.0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        
        losses = -F.logsigmoid(beta * logits)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        ZeRO-3/NCCL-safe loss:
        - keep CE loss from both clean & augmented always in graph
        - add ranking loss as an extra term
        - avoid in-place mutation on inputs
        - no extra gradient-carrying forward for attention (use no_grad)
        """
        aug_info = {}

        if self.use_augment:
            if self.aug_type in ["mergemix", "mergemix-r"]:
                with torch.no_grad():
                    vision_output = model(**inputs, return_attn=self.use_attn)
                aug_img, aug_info = self.augmenter(inputs["images"], vision_output)
            else:
                aug_img, aug_info = self.augmenter(inputs["images"])

            worst_inputs = {k: v for k, v in inputs.items()}
            worst_inputs["images"] = aug_img.to(dtype=inputs["images"].dtype).contiguous()

            if "lam" in aug_info and "index" in aug_info and not getattr(self, "ranking_loss", False):
                idx = aug_info["index"]
                worst_inputs["labels"] = inputs["labels"] if aug_info["lam"] <= 0.5 else inputs["labels"][idx, :]
        else:
            return model(**inputs).loss

        better_out = model(**inputs)  # carries loss & logits
        gt_loss = better_out.loss
        # gt_loss = 0.0

        #  Ranking loss
        if getattr(self, "ranking_loss", False):
            # FIXME
            # Optimization only for Winner
            with torch.no_grad():
                worst_out  = model(**worst_inputs)  # carries loss & logits
            # Optimization for both
            # worst_out  = model(**worst_inputs)  # carries loss & logits

            labels = inputs["labels"]
            shift = self.base_vision_tok - int(getattr(self, "tome_merge_num", 0))
            better_log = better_out.logits[:, shift:, :]

            # FIXME
            worst_log = worst_out.logits[:,  shift:, :].clone().detach().requires_grad_() # restore grad only for simpo_loss computation 
            # worst_log = worst_out.logits[:,  shift:, :]

            seq_len = min(better_log.size(1), worst_log.size(1), labels.size(1))
            if seq_len > 1:
                better_log = better_log[:, :seq_len, :]
                worst_log = worst_log[:, :seq_len, :]
                labels_use = labels[:, :seq_len]
                best_log_prob  = self.get_batch_logps(better_log, labels_use, return_per_token_logp=False)
                worst_log_prob = self.get_batch_logps(worst_log, labels_use, return_per_token_logp=False)

                # the logits shape is [b, labels, 32000(vocab_size)]
                best_length, worst_length = better_log.shape[-2], worst_log.shape[-2]
                lam = float(aug_info.get("lam", 1.0)) if self.use_augment else 0.7
                lam_ = self.lambda_scale(self.tome_merge_num / self.base_vision_tok, lam)
                if self.loss_type == "simpo":
                    rank_loss = self.simpo_loss(
                        best_log_prob,
                        worst_log_prob,
                        best_length,
                        worst_length,
                        beta=1.0,
                        gamma= 1.0 - lam_,
                    )
                elif self.loss_type == "dpo":
                    if self.refer_model is None:
                        rank_loss = 0.0
                    else:
                        with torch.no_grad():
                            refer_better_out = self.refer_model(**inputs)
                            refer_worst_out  = self.refer_model(**worst_inputs)

                        refer_better_log = refer_better_out.logits[:, shift:, :]
                        refer_worst_log = refer_worst_out.logits[:, shift:, :]
                        refer_best_log_prob  = self.get_batch_logps(refer_better_log, labels_use, return_per_token_logp=False)
                        refer_worst_log_prob = self.get_batch_logps(refer_worst_log, labels_use, return_per_token_logp=False)
                        
                        rank_loss, cr, rr = self.dpo_loss(
                            policy_chosen_logps=best_log_prob,
                            policy_rejected_logps=worst_log_prob,
                            reference_chosen_logps=refer_best_log_prob,
                            reference_rejected_logps=refer_worst_log_prob,
                        )
                else:
                    raise ValueError("Wrong ranking loss for preference tuning.")
                gt_loss += rank_loss
                gt_loss = gt_loss / 2
        else:
            gt_loss += model(**worst_inputs).loss
            gt_loss = gt_loss / 2
        if torch.isnan(gt_loss):
            gt_loss = model(**inputs).loss
        assert gt_loss > 0
        return (gt_loss, {"better_out": better_out, "worst_out": worst_out}) if return_outputs else gt_loss


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            # Raw Codes
            # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            
            # SGG Codes
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, model=opt_model)
            if optimizer_cls.__name__ in ["Adam_miniSAC", "Adam_mini"]:
                self.optimizer = optimizer_cls(
                    named_parameters=opt_model.named_parameters(),
                    **optimizer_kwargs
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    ##################################################
    ########## You need to modified some codes in ''traing_args''
    ########## Adding the optimizer name as 'ADAMW_SCALE = adamw_scale', other optimizer same as this.
    ##################################################
    @staticmethod
    def get_optimizer_cls_and_kwargs(
            args, model = None
        ):
            """
            Returns the optimizer class and optimizer parameters based on the training arguments.

            Args:
                args (`transformers.training_args.TrainingArguments`):
                    The training arguments for the training session.

            """

            # parse args.optim_args
            optim_args = {}
            if args.optim_args:
                for mapping in args.optim_args.replace(" ", "").split(","):
                    key, value = mapping.split("=")
                    optim_args[key] = value

            optimizer_kwargs = {"lr": args.learning_rate}
            adam_kwargs = {
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                # "betas": (0.9, 0.99),
                # "eps": 1e-6,
            }
            adan_kwargs = {
                "betas": (0.9, 0.92, 0.99),
                "eps": args.adam_epsilon,
            }
            sgd_kwargs = {
                "momentum": 0.99,
                "weight_decay": 1e-5,
            }
            if args.optim == 'adafactor':
                print('Optimizing based-on Vanilla Adafactor')
                optimizer_cls = Adafactor
                optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
            elif args.optim == 'adamw_hf':
                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim in ['adamw_torch', 'adamw_torch_fused']:
                print('Optimizing based-on Vanilla AdamW')
                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
                if args.optim == 'adamw_torch_fused':
                    optimizer_kwargs.update({"fused": True})
            elif args.optim == 'adamw_sgg':             # AdamW-scale
                print('Optimizing based-on AdamW with SGG')
                optimizer_cls = AdamWSGG
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"n_clusters": 5,
                                         "recluster_interval": 1000,  # defalut setting  
                                         "scale_bound": (1, 10.0), 
                                         "beta3": 0.9
                                        })
            elif args.optim == 'adamw_sgg_v2':             # AdamW-scale version 2
                print('Optimizing based-on AdamW with SGG version_2')
                optimizer_cls = AdamWSGG_v2
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"n_clusters": 3,
                                         "recluster_interval": 1000,  # defalut setting  
                                         "ema_decay_clusters": 0.95, 
                                         "ema_decay_scale": 0.9,
                                        })
            # TODO 
            elif args.optim == 'adamw_sac':
                print('Optimizing based-on AdamW with SAC')
                optimizer_cls = AdamWSAC
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"model": model,
                                         "betas": (0.9, 0.99),
                                         "scale_update_freq": 1000,  # defalut setting  
                                         "scale_bound": (0.5, 10.0) 
                                        })
            elif args.optim == 'adafactor_sgg':             # Adafactor-scale
                print('Optimizing based-on Adafactor with SGG')
                optimizer_cls = AdafactorSGG
                optimizer_kwargs.update({"n_clusters": 3,
                                         "recluster_interval": 1000,  # defalut = 100, we consider that the number of the inter could set as the 1/10.
                                         "ema_decay_clusters": 0.95, 
                                         "ema_decay_scale_factors": 0.9
                                        })
            elif args.optim == 'adafactor_sgg_v2':             # Adafactor-scale
                print('Optimizing based-on Adafactor with SGG version_2')
                optimizer_cls = AdafactorSGG_v2
                optimizer_kwargs.update({"n_clusters": 3,
                                         "recluster_interval": 1000,  # defalut = 100, we consider that the number of the inter could set as the 1/10.
                                         "ema_decay_clusters": 0.95, 
                                         "ema_decay_scale": 0.9
                                        })
            elif args.optim == 'adam_mini':
                print('Optimizing based-on Vanilla Adam-mini')
                optimizer_cls = Adam_mini
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({
                    "dim": 4096,
                    "n_heads": 32,
                })
            elif args.optim == 'adam_mini_sac':
                print('Optimizing based-on Vanilla Adam-mini with SAC')
                optimizer_cls = Adam_miniSAC
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({
                    "betas": (0.9, 0.99),
                    "eps": 1e-8,
                    "dim": 4096,
                    "n_heads": 32,
                    "scale_update_freq": 1000,  # defalut setting  
                    "scale_bound": (0.1, 10.0) 
                })
            elif args.optim == 'lamb':
                print('Optimizing based-on Vanilla Lamb')
                optimizer_cls = Lamb
                # optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({
                    "betas": (0.9, 0.99),
                })
            elif args.optim == 'lamb_sgg':
                print('Optimizing based-on Lamb with SGG')
                optimizer_cls = LambSGG
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"n_clusters": 2,
                                         "recluster_interval": 1000,
                                         "scale_bound": (1, 10.0), 
                                         "beta3": 0.9
                                        })
            elif args.optim == 'lamb_sgg_v2':
                print('Optimizing based-on Lamb with SGG version_2')
                optimizer_cls = LambSGG_v2
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"n_clusters": 2,
                                         "recluster_interval": 1000,
                                         "beta3": 0.999, 
                                         "T_total": 11000
                                        })
            # BUG
            elif args.optim == 'muon':
                print('Optimizing based-on Vanilla Muon')
                optimizer_cls = Muon
                optimizer_kwargs.update({"nesterov": True,
                                         "ns_steps": 5,
                                         "adamw_betas": (args.adam_beta1, args.adam_beta2),
                                         "adamw_eps": args.adam_epsilon,
                                        })
            elif args.optim == 'shampoo':
                print('Optimizing based-on Vanilla Shampoo')
                optimizer_cls = Shampoo
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'shampoo_sac':
                print('Optimizing based-on Vanilla Shampoo with SAC')
                optimizer_cls = ShampooSAC
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"model": model,
                                         "scale_update_freq": 1000,  # defalut setting  
                                         "scale_bound": (0.5, 1.0) 
                                        })
            elif args.optim == 'shampoo_sgg_v2':
                print('Optimizing based-on Vanilla Shampoo with version_2')
                optimizer_cls = ShampooSGG_v2
                optimizer_kwargs = {"lr": args.learning_rate,
                                    "betas": (args.adam_beta1, args.adam_beta2),
                                    "eps": args.adam_epsilon,
                                    "n_clusters": 5,
                                    "recluster_interval": 1000,  # defalut = 100, we consider that the number of the inter could set as the 1/10.
                                    "scale_bound": (0.5, 10.0),
                                    "beta3": 0.9,
                                    "optimize_1d": True,
                                    "lr_1d": args.learning_rate,
                                    }
            elif args.optim == 'soap':
                print('Optimizing based-on Vanilla SOAP')
                optimizer_cls = SOAP
                # optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'mars':
                # [Types]: mars-adamw, mars-shampoo, mars-lion
                mars_type = 'mars-lion'  # mars-adamw, mars-shampoo, mars-lion
                print('Optimizing based-on Vanilla MARS with {}'.format(mars_type))
                optimizer_cls = MARS
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"gamma": 0.025,
                                         "lr_1d": 2e-6,  # 2e-6 for lion
                                         "is_approx": True,
                                         "mars_type": mars_type,
                                         "optimize_1d": False,
                                         "weight_decay_1d": 0.1,
                                         "betas_1d": (0.9, 0.98) # (0.9, 0.98) for lion
                                        })
            elif args.optim == 'radam':
                print('Optimizing based-on Vanilla RAdam')
                optimizer_cls = RAdam
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'nadam':
                print('Optimizing based-on Vanilla NAdam')
                optimizer_cls = NAdam
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'adan':
                print('Optimizing based-on Vanilla Adan')
                optimizer_cls = Adan
                optimizer_kwargs.update(adan_kwargs)
            elif args.optim == 'came':
                print('Optimizing based-on Vanilla CAME')
                optimizer_cls = CAME
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'sgd':
                print('Optimizing based-on Vanilla SGD')
                optimizer_cls = SGD
                optimizer_kwargs.update(sgd_kwargs)
            elif args.optim == 'sophiag':
                print('Optimizing based-on Vanilla SophiaG')
                optimizer_cls = SophiaG
                optimizer_kwargs = {"lr": 2e-6,
                                    "betas": (args.adam_beta1, args.adam_beta2),
                                    }
            elif args.optim == 'lion':
                print('Optimizing based-on Vanilla Lion')
                optimizer_cls = Lion
                optimizer_kwargs = {"lr": 2e-6,
                                    "betas": (0.9, 0.98),
                                    }
            elif args.optim == 'galore_adamw':
                print('Optimizing based-on Vanilla GaLore_AdamW')
                optimizer_cls = GaLore_AdamW
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'conda':
                print('Optimizing based-on Vanilla Conda')
                optimizer_cls = Conda
                optimizer_kwargs.update(adam_kwargs)
            elif args.optim == 'apollo_adamw':
                print('Optimizing based-on Vanilla APOLLO_AdamW')
                optimizer_cls = APOLLO_AdamW
                optimizer_kwargs.update(adam_kwargs)
            else:
                raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
            return optimizer_cls, optimizer_kwargs


    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

        
        # if self.loss_type != "dpo":
        #     return

        # current_step = self.state.global_step
        # if current_step - self.last_reference_step >= self.reference_update_interval and current_step >= 500:
        #     print(f"\n[Step {current_step}] Updating online reference model...")
        #     self.update_reference_model()
        #     self.last_reference_step = current_step


    def update_reference_model(self):
        checkpoint_dir = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        if not os.path.exists(checkpoint_dir):
            print("Checkpoint not found, skip update.")
            return
        try:
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            new_ref = LlavaLlamaForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.model.dtype,
                low_cpu_mem_usage=False,
            ).to(self.model.device)

            new_ref.eval()
            for p in new_ref.parameters():
                p.requires_grad = False

            if self.refer_model is not None:
                del self.refer_model
                torch.cuda.empty_cache()

            self.refer_model = new_ref
            print(f"Online reference model updated from {checkpoint_dir}")

        except Exception as e:
            print(f"Failed to update reference model: {e}")


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
