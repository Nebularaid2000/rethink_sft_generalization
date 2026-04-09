# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os
import math

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.sft_dataset import OurSFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup, get_constant_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.trainer.ppo.core_algos import agg_loss
import verl.utils.torch_functional as verl_F
import wandb

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device

        # entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        # self.compute_entropy_from_logits = (
        #     torch.compile(entropy_from_logits, dynamic=True)
        #     if self.config.get("use_torch_compile", True)  #  use torch compile by default
        #     else entropy_from_logits
        # )

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=config.data.shuffle_train, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if "Ministral-3" in self.config.model.partial_pretrain:
            from transformers import Mistral3Config, Mistral3ForConditionalGeneration, Ministral3Config, Ministral3ForCausalLM
        if "Qwen2.5-VL" in self.config.model.partial_pretrain:
            from transformers import Qwen2_5_VLForConditionalGeneration

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        # if "Ministral-3" in self.config.model.partial_pretrain:
        #     print("using Ministral-3 config!!")
        #     config = Mistral3Config.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        # else:
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        # if hasattr(self.model_config, "max_position_embeddings"):
        #     self.model_config.max_position_embeddings = max(
        #         self.model_config.max_position_embeddings, self.config.data.max_length
        #     )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )
        # print(f"self.config.model.partial_pretrain: {self.config.model.partial_pretrain}")
        with init_context():
            if "Ministral-3" in self.config.model.partial_pretrain:
                self.model: PreTrainedModel = Mistral3ForConditionalGeneration.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )
            else:
                config.attn_implementation = "flash_attention_2"
                print(f"after modify fsdp trainer config.attn_implementation: {config.attn_implementation}")
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            auto_wrap_policy = get_fsdp_wrap_policy(
                self.model,
                config=self.config.model.fsdp_config.wrap_policy,
                is_lora=self.config.model.get("lora_rank", 0) > 0,
            )
            if self.device_mesh.get_rank() == 0:
                print(auto_wrap_policy)
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        total_epochs = float(self.config.trainer.total_epochs)
        default_total_steps = self.steps_per_epoch * total_epochs
        configured_total_steps = (
            self.config.trainer.total_training_steps
            if self.config.trainer.total_training_steps is not None
            else default_total_steps
        )
        # Use ceil to avoid under-training when total_epochs is fractional (e.g. 1.5 epoch).
        self.total_steps = int(math.ceil(configured_total_steps))

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        
        loss_mask = batch.pop("loss_mask")[:, :-1].to(self.device_name)
        advantages = batch.pop("advantages").to(self.device_name)

        ref_log_prob = batch.get("ref_log_prob", None)
        if ref_log_prob is not None:
            ref_log_prob = ref_log_prob[:, :-1].to(self.device_name) # should align with loss_mask, so use [:, :-1] to shift one position
        
        mean_ref_log_prob = batch.get("mean_ref_log_prob", None)
        if mean_ref_log_prob is not None:
            mean_ref_log_prob = mean_ref_log_prob.to(self.device_name)
        
        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous() # (bsz, max_seqlen - 1, vocab_size)
                shift_labels = labels.contiguous() # (bsz, max_seqlen - 1)
                # Flatten the tokens
                # shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                # shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                if ref_log_prob is not None:
                    ref_log_prob = ref_log_prob.to(shift_logits.device)

                with torch.no_grad():
                    flattened_logits = shift_logits.flatten(end_dim=-2)[loss_mask.flatten().bool()]
                    entropy = verl_F.entropy_from_logits_with_chunking(flattened_logits, chunk_size=512)
                    mean_entropy = entropy.mean().item()
                del flattened_logits, entropy
                torch.cuda.empty_cache()

                log_prob = verl_F.logprobs_from_logits(shift_logits, labels=shift_labels)                

                advantages = torch.ones_like(log_prob).detach() * advantages # seq-level -> token-level advantages

                clip_ratio = self.config.trainer.clip_ratio
                clip_ratio_low = (
                    self.config.trainer.clip_ratio_low if self.config.trainer.clip_ratio_low is not None else clip_ratio
                )
                clip_ratio_high = (
                    self.config.trainer.clip_ratio_high if self.config.trainer.clip_ratio_high is not None else clip_ratio
                )
                
                del logits, shift_logits
                torch.cuda.empty_cache()

                loss, ratio, ratio_clip, clip_frac_low, clip_frac_high = \
                    compute_loss(log_prob, 
                                response_mask=loss_mask,
                                advantages=advantages, 
                                old_log_prob=ref_log_prob,
                                mean_old_log_prob=mean_ref_log_prob,
                                imp_sampling_mode=self.config.trainer.importance_sampling_mode,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                loss_agg_mode=self.config.trainer.loss_agg_mode)
                
                if do_backward:
                    loss.backward()
                
                loss_item = loss.item()
                if ref_log_prob is not None:
                    del ref_log_prob
                del loss, log_prob, advantages, input_ids, attention_mask, position_ids
                torch.cuda.empty_cache()
                

            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                raise NotImplementedError(
                    "Sequence parallelism with remove padding is not implemented yet in this script. "
                    "Please set use_remove_padding=False or ulysses_sequence_parallel_size=1."
                )          
            
            return loss_item, mean_entropy, ratio, ratio_clip, clip_frac_low, clip_frac_high

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        step_entropy = 0
        clip_frac_low_mean = 0
        clip_frac_high_mean = 0
        ratio_list = []
        ratio_clip_list = []
        for micro_batch in micro_batches:
            loss, mean_entropy, ratio, ratio_clip, clip_frac_low, clip_frac_high = self._compute_loss_and_backward(batch=micro_batch)
            step_loss += loss / n_micro_batches
            step_entropy += mean_entropy / n_micro_batches
            clip_frac_low_mean += clip_frac_low / n_micro_batches
            clip_frac_high_mean += clip_frac_high / n_micro_batches
            ratio_list.extend(ratio)
            ratio_clip_list.extend(ratio_clip)
        
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        step_entropy = torch.tensor(step_entropy).to(self.device_name)
        grad_norm = torch.tensor(grad_norm.item()).to(self.device_name)
        clip_frac_low_mean = torch.tensor(clip_frac_low_mean).to(self.device_name)
        clip_frac_high_mean = torch.tensor(clip_frac_high_mean).to(self.device_name)

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(step_entropy, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(clip_frac_low_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(clip_frac_high_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(grad_norm, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
            torch.distributed.all_reduce(step_entropy)
            step_entropy /= self.device_mesh.size(0)
            torch.distributed.all_reduce(grad_norm)
            grad_norm /= self.device_mesh.size(0)
            torch.distributed.all_reduce(clip_frac_low_mean)
            clip_frac_low_mean /= self.device_mesh.size(0)
            torch.distributed.all_reduce(clip_frac_high_mean)
            clip_frac_high_mean /= self.device_mesh.size(0)

        # Gather ratio lists from all processes
        all_ratios = [None for _ in range(self.device_mesh.size(0))]
        all_ratios_clip = [None for _ in range(self.device_mesh.size(0))]
        torch.distributed.all_gather_object(all_ratios, ratio_list)
        torch.distributed.all_gather_object(all_ratios_clip, ratio_clip_list)
        if self.device_mesh.get_rank() == 0:
            # Flatten all ratios from all processes
            ratio_list_all = [r for ratios in all_ratios for r in ratios]
            ratio_clip_list_all = [r for ratios in all_ratios_clip for r in ratios]
            mean_ratio = sum(ratio_list_all) / len(ratio_list_all) if ratio_list_all else 0.0
            mean_ratio_clip = sum(ratio_clip_list_all) / len(ratio_clip_list_all) if ratio_clip_list_all else 0.0
            print(f"ratio_list_all type: {type(ratio_list_all)}")
            print(f"ratio_list_all length: {len(ratio_list_all)}")
        
        metric = {
            "train/loss": step_loss.detach().item(), 
            "train/lr": lr,
            "train/grad_norm": grad_norm.detach().item(),
            "train/clip_frac_low": clip_frac_low_mean.detach().item(),
            "train/clip_frac_high": clip_frac_high_mean.detach().item(),
            "train/entropy/mean": step_entropy.detach().item(),
        }
        if self.device_mesh.get_rank() == 0: # only record on rank 0
            metric["train/ratio/mean"] = mean_ratio
            metric["train/ratio/dist"] = ratio_list_all
            metric["train/ratio_clip/mean"] = mean_ratio_clip
            metric["train/ratio_clip/dist"] = ratio_clip_list_all
        return metric

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            # temp_tracker_file = tracker_file + ".tmp"
            with open(tracker_file, "w") as f:
                f.write(str(step))
            # os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def _create_histogram(self, logs_dict, key):
        data = logs_dict.pop(key, None)
        if data is not None:            
            hist = wandb.Histogram(data)
            return hist
        return None

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = self.total_steps
        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        epoch = start_epoch
        while global_step < self.total_training_steps:
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                if rank == 0: # Log only on rank 0, so some gather operations can stay on rank 0 as well.
                    # Build a histogram for the ratio distribution using wandb directly.
                    for hist_key in ["train/ratio/dist", "train/ratio_clip/dist"]:
                        hist = self._create_histogram(metric, hist_key)
                        if hist is not None:
                            wandb.log(data={hist_key: hist}, step=global_step)
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # TODO: We discard the eval step for simplicity
                # # early exit or validation step
                # if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                #     # Perform validation
                #     val_losses = []
                #     for val_data in self.val_dataloader:
                #         val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                #             self.device_name
                #         )
                #         val_loss = self.validation_step(val_data)
                #         val_losses.append(val_loss)
                #     if rank == 0:
                #         val_loss = torch.mean(torch.stack(val_losses))
                #         metric = {"val/loss": val_loss.detach().item()}
                #         tracking.log(data=metric, step=global_step)
                #         last_valid_metric = metric
                #     torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return
            epoch += 1


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    if config.data.eos_token is not None:
        tokenizer.eos_token = config.data.eos_token
    if config.data.eos_token_id is not None:
        tokenizer.eos_token_id = config.data.eos_token_id
    
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        # dataset_cls = SFTDataset
        dataset_cls = OurSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


def clip_fn(ratio, cliprange_low, cliprange_high, response_mask):
    """
    Clip the ratio to the specified range.
    
    Args:
        ratio (torch.Tensor): The ratio tensor to be clipped. Note: ratio may have been detached from the computation graph.
        cliprange_low (float): The lower bound for clipping.
        cliprange_high (float): The upper bound for clipping.
    
    Returns:
        torch.Tensor: The clipped ratio tensor.
    """
    if cliprange_low is not None and cliprange_high is not None:
        ratio_clip = torch.clamp(ratio, min=1.0 - cliprange_low, max=1.0 + cliprange_high)
        clip_frac_low = verl_F.masked_mean((ratio < 1.0 - cliprange_low).float(), response_mask).item()
        clip_frac_high = verl_F.masked_mean((ratio > 1.0 + cliprange_high).float(), response_mask).item()
    else:
        ratio_clip = ratio
        clip_frac_low = 0.0
        clip_frac_high = 0.0

    return ratio_clip, clip_frac_low, clip_frac_high


def compute_loss(
    log_prob,
    response_mask,
    advantages = None,
    old_log_prob = None,
    mean_old_log_prob = None,
    imp_sampling_mode: str = "none",
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    ratio_for_logging = []
    ratio_clip_for_logging = []
    clip_frac_low = 0.0
    clip_frac_high = 0.0

    if imp_sampling_mode == "vanilla":
        losses = -log_prob # token-level loss

        if old_log_prob is not None:
            # the following ratio statistics is only for logging
            negative_approx_kl = log_prob - old_log_prob
            # Clamp negative_approx_kl for stability
            negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl).detach()
            ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask)
            ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
            ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()


    elif imp_sampling_mode == "teacher-weighted": # -π_teacher * log_prob
        weight = torch.exp(old_log_prob).detach()
        losses = -weight * log_prob # token-level loss

        if old_log_prob is not None:
            # the following ratio statistics is only for logging
            negative_approx_kl = log_prob - old_log_prob
            # Clamp negative_approx_kl for stability
            negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl).detach()
            ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask)
            ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
            ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
    

    elif imp_sampling_mode == "dft":
        weight = torch.exp(log_prob).detach() # sg(π_θ(yi|x))
        weight_clip, clip_frac_low, clip_frac_high = clip_fn(weight, cliprange_low, cliprange_high, response_mask)
        losses = -weight_clip * log_prob
        ratio_for_logging = weight.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = weight_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
    

    elif imp_sampling_mode == "adv-only": # Compare with vanilla to test the effect of negative samples.
        assert advantages is not None
        assert log_prob.shape == advantages.shape, (
            "log_prob and advantages must have the same shape when using adv-only importance sampling"
        )
        losses = -advantages * log_prob

        if old_log_prob is not None:
            negative_approx_kl = log_prob - old_log_prob
            # Clamp negative_approx_kl for stability
            negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl).detach()
            ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask)
            ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
            ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()


    elif imp_sampling_mode == "ppo-ori": # PPO-style importance sampling (token-level) with original clipping; loss may appear negative, but its sign is unrelated to gradient direction.
        assert old_log_prob is not None and advantages is not None
        assert log_prob.shape == old_log_prob.shape == advantages.shape, (
            "log_prob, old_log_prob, and advantages must have the same shape when using ppo importance sampling"
        )
        negative_approx_kl = log_prob - old_log_prob
        # Clamp negative_approx_kl for stability
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        pg_losses1 = -advantages * ratio
        ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask) # direct clipping on ratio (may zero out gradients)
        pg_losses2 = -advantages * ratio_clip
        losses = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)

        ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
    

    elif imp_sampling_mode == "ppo": # The final gradient takes the form (pi_theta / pi_old) * log_prob.
        assert old_log_prob is not None and advantages is not None
        assert log_prob.shape == old_log_prob.shape == advantages.shape, (
            "log_prob, old_log_prob, and advantages must have the same shape when using ppo importance sampling"
        )
        negative_approx_kl = log_prob - old_log_prob
        # Clamp negative_approx_kl for stability
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl) # Compared with cispo, this mainly differs by a detach.

        ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask) # direct clipping on ratio (may zero out gradients)
        losses = -advantages * ratio_clip

        ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()


    elif imp_sampling_mode == "clip": # Clip tokens with overly large ratio = pi_theta / pi_old, while keeping gradients in log_prob form.
        assert old_log_prob is not None and advantages is not None
        assert log_prob.shape == old_log_prob.shape == advantages.shape, (
            "log_prob, old_log_prob, and advantages must have the same shape when using ppo importance sampling"
        )
        negative_approx_kl = log_prob - old_log_prob
        # Clamp negative_approx_kl for stability
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).detach()
        
        ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask) # direct clipping on ratio (may zero out gradients)

        clip_mask = torch.ones_like(ratio)
        clip_mask[ratio > 1.0 + cliprange_high] = 0.0
        clip_mask[ratio < 1.0 - cliprange_low] = 0.0

        losses = -advantages * log_prob * clip_mask

        ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()


    elif imp_sampling_mode == "cispo": # CISPO-style importance sampling (token-level)
        assert old_log_prob is not None and advantages is not None
        assert log_prob.shape == old_log_prob.shape == advantages.shape, (
            "log_prob, old_log_prob, and advantages must have the same shape when using ppo importance sampling"
        )
        negative_approx_kl = log_prob - old_log_prob
        # Clamp negative_approx_kl for stability
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).detach()

        ratio_clip, clip_frac_low, clip_frac_high = clip_fn(ratio, cliprange_low, cliprange_high, response_mask) # clipping on detached ratio (only clip importance ratio, not log_prob)
        losses = -advantages * ratio_clip * log_prob # since ratio_clip is detached, we need to use logprob here rather than prob

        ratio_for_logging = ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()


    # https://github.com/volcengine/verl/pull/2775
    elif imp_sampling_mode == "gspo-ori": # GSPO-style importance sampling (seq-level/token-level)
        assert old_log_prob is not None and advantages is not None
        negative_approx_kl = log_prob - old_log_prob

        # compute sequence-level importance ratio:
        # si(θ) = (π_θ(yi|x)/π_old(yi|x))^(1/|yi|) =
        # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

        # Combined ratio at token level:
        # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
        # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
        log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

        # finaly exp() to remove log
        seq_importance_ratio = torch.exp(log_seq_importance_ratio)

        pg_losses1 = -advantages * seq_importance_ratio
        seq_importance_ratio_clip, clip_frac_low, clip_frac_high = clip_fn(seq_importance_ratio, cliprange_low, cliprange_high, response_mask)
        pg_losses2 = -advantages * seq_importance_ratio_clip
        losses = torch.maximum(pg_losses1, pg_losses2)

        # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)

        ratio_for_logging = seq_importance_ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = seq_importance_ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
    

    elif imp_sampling_mode == "gspo":
        assert old_log_prob is not None and advantages is not None
        negative_approx_kl = log_prob - old_log_prob

        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

        log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

        seq_importance_ratio = torch.exp(log_seq_importance_ratio)

        seq_importance_ratio_clip, clip_frac_low, clip_frac_high = clip_fn(seq_importance_ratio, cliprange_low, cliprange_high, response_mask)
        losses = -advantages * seq_importance_ratio_clip

        ratio_for_logging = seq_importance_ratio.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()
        ratio_clip_for_logging = seq_importance_ratio_clip.flatten()[response_mask.flatten().bool()].detach().cpu().tolist()

    else:
        raise ValueError(
            f"Invalid importance sampling mode: {imp_sampling_mode}. "
        )

    # do mask and aggregation across tokens
    loss = agg_loss(loss_mat=losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return loss, ratio_for_logging, ratio_clip_for_logging, clip_frac_low, clip_frac_high


if __name__ == "__main__":
    main()
