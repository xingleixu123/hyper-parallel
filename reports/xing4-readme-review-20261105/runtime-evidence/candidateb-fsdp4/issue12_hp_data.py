"""Single-rank issue12 Ordered data integration for HP trainer."""
import torch
from ordered_factory import OrderedFourFieldDataset, collate_ordered

IGNORE_INDEX = -100


def build_dataset(*, prefixes, sequence_length, pad_token, data_config=None, transform=None, tokenizer=None,
                  mesh_context=None, training_config=None):
    """Build the unshuffled dataset; HP's native batch sampler owns its cursor."""
    del data_config, transform, tokenizer, mesh_context, training_config
    return OrderedFourFieldDataset(prefixes, sequence_length=sequence_length, pad_token=pad_token)


def build_collator(*, mesh_context=None):
    """Convert MF mask to the labels expected by TextParallelBatch."""
    del mesh_context
    return collate_for_text


def collate_for_text(samples):
    """Retain masked-token semantics after Ordered pad labels have become zero."""
    batch = collate_ordered(samples)
    if not torch.all((batch["loss_mask"] == 0) | (batch["loss_mask"] == 1)):
        raise ValueError("Ordered loss mask must be binary")
    batch["labels"] = batch["labels"].masked_fill(batch["loss_mask"] == 0, IGNORE_INDEX)
    batch["tokens"] = batch.pop("input_ids")
    return batch
