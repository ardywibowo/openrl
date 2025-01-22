import time
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import (  # todo: what is seed worker?
    seed_worker, speed_metrics)

from treetune.common.logging_utils import get_logger
from treetune.trainers import Trainer
from treetune.trainers.mle_trainer import MaximumLikelihoodTrainer
from treetune.trainers.policy_trainer import PolicyTrainer
from treetune.trainers.utils import entropy_from_logits, masked_mean

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    from transformers.utils.hub import cached_file

    is_flash_attn_xentropy_available = True
except ImportError:
    from torch.nn import CrossEntropyLoss

    is_flash_attn_xentropy_available = (
        True  # todo: why is this set to True? and why is it not used?
    )

logger = get_logger(__name__)


class BinaryClassificationDataCollator:
    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates the given data instances into a batch.
        Every data instance should have the following keys:
        - "query_token_ids": The token ids of the query.
        - "target_prob": The target probability of the query.

        Args:
            data_instances (List[Dict[str, Any]]):
                The data instances to collate.
        Returns:
            Dict[str, torch.Tensor]:
                The collated batch.
                It contains the following keys:
                - "input_ids": The token ids of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "target_prob": The target probability of the query.
        """

        # Get the maximum sequence length
        max_seq_len = max(len(instance["query_token_ids"]) for instance in data_instances)

        # Create the batch
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "target_prob": [],
        }

        # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_token_id = 0

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]

            # Create the input ids and attention mask
            input_ids = query_token_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_len - len(input_ids)

            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["target_prob"].append(instance["target_prob"])

        # Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch


@Trainer.register("binary_classification")
class BinaryClassificationTrainer(MaximumLikelihoodTrainer):
    def __init__(
        self, 
        finetune_iterations: int = 0,
        **kwargs
    ):
        super().__init__(data_collator=BinaryClassificationDataCollator(), **kwargs)
        
        self.finetune_iterations = finetune_iterations
        self.iteration_idx = 0

    def step(self, episodes_dataset: Dataset) -> None:
        
        if self.iteration_idx < self.finetune_iterations:
            logger.info(f"Fine-tuning value head first")
            
            value_head_name = "value_head"
            for name, param in self.model.named_parameters():
                if value_head_name in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            logger.info(f"Fine-tuning the entire model")
            for param in self.model.parameters():
                param.requires_grad = True
        
        super().step(episodes_dataset)
        self.iteration_idx += 1

    def compute_loss(
        self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss for the given inputs. The loss is basically the negative log likelihood
        that weighted by the advantages of the responses.
        Args:
            model (PreTrainedModel): The model to compute the loss for.
            inputs (Dict[str, torch.Tensor]):
                The inputs to the model.
                It should contain the following keys:
                - "input_ids": The token ids of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "target_prob": The target probability of the query.
        Returns:
            torch.Tensor: The loss.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        target_prob: torch.Tensor = inputs["target_prob"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        # Compute the logits
        if not self.is_flash_attention_model:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,  # We don't need the cache for training
            )
        else:
            # Flash attention models do not support attention mask
            # But, since we're padding on right and the model is causal,
            # we're fine without the attention mask.
            assert torch.all(
                attention_mask[:, 0] == 1
            ), "Flash attention models do not support left padding"
            outputs = model(input_ids=input_ids)
        logits = outputs['value']  # Shape: (batch_size, max_seq_len, vocab_size)
        orig_dtype = logits.dtype
        
        # Last indexes is the sum of the attention mask
        last_indexes = (attention_mask.sum(dim=1) - 1).view(-1)

        # Compute the loss in full precision
        logits = logits.to(torch.float32)

        # Extract the last logits according to the last indexes
        last_logits = logits[range(logits.size(0)), last_indexes.long(), :]

        # Flatten the tokens
        last_logits = last_logits.view(-1)
        target_prob = target_prob.view(-1)
        
        mean_entropy = masked_mean(entropy_from_logits(last_logits), torch.ones_like(last_logits))
        mean_entropy = mean_entropy.detach().clone()
        
        loss = F.binary_cross_entropy_with_logits(last_logits, target_prob.to(logits.dtype), reduction="mean")

        loss = loss.to(orig_dtype)

        metrics = {"value_head_logit_entropy": mean_entropy}

        return loss, metrics

    def _create_dataloader(self, episodes_dataset: Dataset) -> DataLoader:
        if episodes_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        logger.error(f"Before creating data loader")

        data_collator = self.data_collator
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        sampler = RandomSampler(episodes_dataset, generator=generator)

        data_loader = DataLoader(
            dataset=episodes_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            sampler=sampler,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=seed_worker,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )

        logger.error(f"After creating data loader")

        return self.accelerator.prepare(data_loader)
