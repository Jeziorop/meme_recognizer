"""PyTorch Residual MLP architecture for Meme Gesture Classification + ONNX & CUDA binary exporter."""

from __future__ import annotations
import struct
from pathlib import Path
import torch
import torch.nn as nn

from ml.features import FEATURE_DIM, MEME_CLASSES


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(self.drop(h)))
        return self.act2(x + h)


class MemePoseClassifier(nn.Module):
    def __init__(
        self,
        in_features: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_classes: int = len(MEME_CLASSES),
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.stem_fc = nn.Linear(in_features, hidden_dim)
        self.stem_ln = nn.LayerNorm(hidden_dim)
        self.stem_act = nn.GELU()

        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem_act(self.stem_ln(self.stem_fc(x)))
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)


def export_model_artifacts(
    model: MemePoseClassifier,
    onnx_path: Path,
    cuda_bin_path: Path,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_bin_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_model = model.to("cpu").eval()
    dummy_input = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)

    torch.onnx.export(
        cpu_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pose_features"],
        output_names=["logits"],
    )

    layers = [
        cpu_model.stem_fc.weight, cpu_model.stem_fc.bias,
        cpu_model.stem_ln.weight, cpu_model.stem_ln.bias,
        cpu_model.res1.fc1.weight, cpu_model.res1.fc1.bias,
        cpu_model.res1.ln1.weight, cpu_model.res1.ln1.bias,
        cpu_model.res1.fc2.weight, cpu_model.res1.fc2.bias,
        cpu_model.res1.ln2.weight, cpu_model.res1.ln2.bias,
        cpu_model.res2.fc1.weight, cpu_model.res2.fc1.bias,
        cpu_model.res2.ln1.weight, cpu_model.res2.ln1.bias,
        cpu_model.res2.fc2.weight, cpu_model.res2.fc2.bias,
        cpu_model.res2.ln2.weight, cpu_model.res2.ln2.bias,
        cpu_model.head.weight, cpu_model.head.bias,
    ]
    with open(cuda_bin_path, "wb") as f:
        f.write(struct.pack("<5I", 0x4D454D45, 1, cpu_model.in_features, cpu_model.hidden_dim, cpu_model.num_classes))
        for tensor in layers:
            arr = tensor.detach().cpu().contiguous().numpy().astype("float32")
            f.write(arr.tobytes())
