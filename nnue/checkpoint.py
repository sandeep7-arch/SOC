# nnue/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


CHECKPOINT_VERSION = "1.0"


def save_checkpoint(
    model,
    optimizer,
    path: str | Path,
    epoch: int = 0,
    step: int = 0,
    scheduler=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save complete training state.

    Includes:
        - model weights
        - optimizer state
        - scheduler state
        - epoch
        - step
        - metadata
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "step": step,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "metadata":
            metadata or {},
    }

    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model=None,
    optimizer=None,
    scheduler=None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load checkpoint.

    Optionally restores:
        model
        optimizer
        scheduler

    Returns raw checkpoint dict.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        model is not None
        and "model_state_dict"
        in checkpoint
    ):
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict"
        in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    return checkpoint


def resume_training_state(
    path: str | Path,
    model,
    optimizer,
    scheduler=None,
    device: str = "cpu",
):
    """
    Restore training state.

    Returns:
        epoch
        step
        metadata
    """

    checkpoint = load_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    epoch = checkpoint.get(
        "epoch",
        0,
    )

    step = checkpoint.get(
        "step",
        0,
    )

    metadata = checkpoint.get(
        "metadata",
        {},
    )

    return (
        epoch,
        step,
        metadata,
    )


def checkpoint_info(
    path: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Inspect checkpoint
    without loading model.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    return {
        "version":
            checkpoint.get(
                "version"
            ),
        "epoch":
            checkpoint.get(
                "epoch"
            ),
        "step":
            checkpoint.get(
                "step"
            ),
        "metadata":
            checkpoint.get(
                "metadata",
                {},
            ),
    }


def latest_checkpoint(
    directory: str | Path,
):
    """
    Find newest checkpoint.
    """

    directory = Path(directory)

    checkpoints = sorted(
        directory.glob("*.pt")
    )

    if not checkpoints:
        return None

    return checkpoints[-1]
