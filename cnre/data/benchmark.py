from typing import Optional, Tuple

import torch


def get_dataloaders(
    dataset: torch.utils.data.TensorDataset,
    training_batch_size: int = 50,
    validation_fraction: float = 0.1,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return dataloaders for training and validation.

    Args:
        dataset: holding all theta and x, optionally masks.
        training_batch_size: training arg of inference methods.
        resume_training: Whether the current call is resuming training so that no
            new training and validation indices into the dataset have to be created.
        dataloader_kwargs: Additional or updated kwargs to be passed to the training
            and validation dataloaders (like, e.g., a collate_fn).

    Returns:
        Tuple of dataloaders for training and validation.

    """

    # Get total number of training examples.
    num_examples = len(dataset)

    # Select random train and validation splits from (theta, x) pairs.
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples

    permuted_indices = torch.randperm(num_examples)
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    # Create training and validation loaders using a subset sampler.
    # Intentionally use dicts to define the default dataloader args
    # Then, use dataloader_kwargs to override (or add to) any of these defaults
    # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
    train_loader_kwargs = {
        "batch_size": min(training_batch_size, num_training_examples),
        "drop_last": True,
        "sampler": torch.utils.data.SubsetRandomSampler(train_indices.tolist()),
    }
    val_loader_kwargs = {
        "batch_size": min(training_batch_size, num_validation_examples),
        "shuffle": False,
        "drop_last": True,
        "sampler": torch.utils.data.SubsetRandomSampler(val_indices.tolist()),
    }
    if dataloader_kwargs is not None:
        train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
        val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset, **train_loader_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset, **val_loader_kwargs)

    return train_loader, val_loader
