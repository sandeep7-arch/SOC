# nnue/export.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn

from .config import CONFIG


class NNUEExporter:
    """
    Export utilities for NNUE models.

    Supports:
        - Engine checkpoints (Inference weights only)
        - Dynamic quantization (INT8 Linear matrix transformations)
        - ONNX export (Graph tracing with dynamic axes)
        - CPU optimization (TorchScript compilation)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    # =====================================================
    # Engine Checkpoint Export
    # =====================================================

    def export_engine_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """
        Save only inference weights.
        Strips out all optimizer metrics, schedulers, and meta arrays.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "export_type": "engine",
                "topology": "HalfKP",
                "feature_dim": CONFIG.FEATURE_DIM,
            },
            path,
        )
        print(f" -> Pure evaluation checkpoint serialized to: {path}")

    # =====================================================
    # CPU Optimization
    # =====================================================

    def optimize_for_cpu(self) -> nn.Module | torch.jit.ScriptModule:
        """
        Compile and optimize the model graph for low-latency CPU execution.
        """
        model = self.model.eval()

        try:
            print(" -> Tracing runtime operational tree via TorchScript...")
            scripted = torch.jit.script(model)
            scripted = torch.jit.optimize_for_inference(scripted)
            return scripted
        except Exception as e:
            print(f" ⚠️  TorchScript compilation skipped due to dynamic list handling: {e}")
            print(" -> Returning standard baseline evaluation model mapping.")
            return model

    # =====================================================
    # Dynamic Quantization
    # =====================================================

    def quantize_model(self) -> nn.Module:
        """
        Apply dynamic INT8 quantization to all dense Linear matrix layers.
        Significantly cuts down execution footprint while preserving score precision.
        """
        print(" -> Compressing matrix nodes using dynamic INT8 quantization...")
        self.model.eval()

        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec={nn.Linear},  # Specifically target L2, L3 deep linear evaluation steps
            dtype=torch.qint8
        )
        return quantized_model

    # =====================================================
    # ONNX Export Engine
    # =====================================================

    def export_onnx(
        self,
        path: str | Path,
        batch_size: int = 1,
    ) -> None:
        """
        Serialize the evaluation network graph into standard open format.
        Assumes dynamic batching arrays using standardized uniform input indices.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Generate standard static padded inputs to map the graph topology cleanly (max 32 pieces)
        dummy_w_feats = torch.zeros((batch_size, 32), dtype=torch.long, device=self.device)
        dummy_b_feats = torch.zeros((batch_size, 32), dtype=torch.long, device=self.device)
        dummy_turns = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        # Pack dummy structures exactly how your model's forward step extracts inputs
        dummy_inputs = (dummy_w_feats, dummy_b_feats, dummy_turns)

        print(f" -> Generating ONNX tensor structure map (Batch size fallback: {batch_size})...")

        try:
            torch.onnx.export(
                self.model,
                dummy_inputs,
                str(path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["white_features", "black_features", "side_to_move"],
                output_names=["position_evaluation"],
                dynamic_axes={
                    "white_features": {0: "batch_size"},
                    "black_features": {0: "batch_size"},
                    "side_to_move": {0: "batch_size"},
                    "position_evaluation": {0: "batch_size"}
                }
            )
            print(f" -> ONNX graph successfully exported to: {path}")
        except Exception as e:
            print(f" ⚠️  ONNX direct trace halted: {e}")
            print(" -> Note: For custom sparse layers, ensure your model's forward path accepts "
                  "uniform padded tensors when executing under ONNX runtimes.")
