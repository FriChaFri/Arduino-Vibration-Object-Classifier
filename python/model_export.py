"""Skeleton for exporting trained models to a Teensy-friendly format."""
from __future__ import annotations

from pathlib import Path


class ModelExporter:
    def __init__(self, model_path: Path, output_header: Path):
        self.model_path = Path(model_path)
        self.output_header = Path(output_header)

    def export(self):
        """Convert the trained Python model to a lightweight representation for firmware."""
        raise NotImplementedError("Implement quantization / parameter export and emit C headers")
