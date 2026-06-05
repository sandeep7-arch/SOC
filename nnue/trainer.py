# nnue/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint, resume_training_state
from .loss import HuberValueLoss
from .config import CONFIG


class NNUETrainer:
    """
    Production-grade NNUE trainer.
    Handles dual-perspective feature translation and high-performance training state.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        device: Optional[str] = None,
    ) -> None:

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=CONFIG.LEARNING_RATE,
            weight_decay=CONFIG.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=CONFIG.NUM_EPOCHS,
        )

        self.loss_fn = HuberValueLoss()

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == "cuda")
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ==================================================
    # Dataloaders
    # ==================================================

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

    def val_loader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

    # ==================================================
    # Fixed Collate Function
    # ==================================================

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Unpacks individual components from NNUEDataset's tuple structures."""
        w_feats, b_feats, turns, targets, weights = zip(*batch)

        return {
            "w_feats": list(w_feats),
            "b_feats": list(b_feats),
            "turns": torch.stack(turns),
            "targets": torch.stack(targets),
            "weights": torch.stack(weights),
        }

    # ==================================================
    # Training Step
    # ==================================================

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        loader = self.train_loader()

        for batch in loader:
            # Move list tensors to device individually to respect varying sparse piece lengths
            w_feats = [x.to(self.device) for x in batch["w_feats"]]
            b_feats = [x.to(self.device) for x in batch["b_feats"]]

            turns = batch["turns"].to(self.device)
            targets = batch["targets"].to(self.device)
            weights = batch["weights"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # AMP Autocast block for accelerated FP16 execution matrix operations
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                # Leverage our actual perspective tracking method
                predictions = self.model.evaluate_batch(
                    w_feats, b_feats, turns, self.device
                )

                loss = self.loss_fn(predictions, targets, weights)

            # Scaled Backward step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Enforce Gradient Clipping Bounds to keep training steps stable
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                CONFIG.GRAD_CLIP_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1

        return total_loss / max(1, len(loader))

    # ==================================================
    # Validation Step
    # ==================================================

    @torch.no_grad()
    def validate(self) -> float:
        loader = self.val_loader()
        if loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            w_feats = [x.to(self.device) for x in batch["w_feats"]]
            b_feats = [x.to(self.device) for x in batch["b_feats"]]

            turns = batch["turns"].to(self.device)
            targets = batch["targets"].to(self.device)
            weights = batch["weights"].to(self.device)

            predictions = self.model.evaluate_batch(
                w_feats, b_feats, turns, self.device
            )

            loss = self.loss_fn(predictions, targets, weights)
            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    # ==================================================
    # Run Orchestrator Pipeline
    # ==================================================

    def train(
        self,
        epochs: int = CONFIG.NUM_EPOCHS,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1,
    ) -> None:

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step()

            print(
                f"[Epoch {epoch+1:02d}/{epochs:02d}] "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

            # Best validation checkpoint serialization guard
            if self.val_dataset and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    path=checkpoint_dir / "best.pt",
                    epoch=epoch,
                    step=self.global_step,
                    metadata={"val_loss": val_loss},
                )

            # Periodic backup storage check
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    path=checkpoint_dir / f"epoch_{epoch+1}.pt",
                    epoch=epoch,
                    step=self.global_step,
                )

    def resume(self, checkpoint_path: str | Path) -> None:
        epoch, step, _ = resume_training_state(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            str(self.device),
        )
        self.start_epoch = epoch + 1
        self.global_step = step

    def save_final(self, path: str | Path) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=path,
            epoch=self.start_epoch,
            step=self.global_step,
        )
