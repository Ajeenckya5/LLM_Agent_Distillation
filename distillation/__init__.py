"""Distillation package: Grok-4 teacher annotation, QLoRA training, student inference."""
from .grok_teacher import annotate_traces
from .qlora_trainer import train, TrainingConfig
from .student_analyzer import StudentAnalyzer

__all__ = ["annotate_traces", "train", "TrainingConfig", "StudentAnalyzer"]
