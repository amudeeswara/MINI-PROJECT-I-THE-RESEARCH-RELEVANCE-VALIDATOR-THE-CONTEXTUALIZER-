"""The Contextualizer — a Research Relevance Validator."""
from .models import Document, Evidence, ValidationResult
from .validator import DEFAULT_WEIGHTS, RelevanceValidator, validate

__version__ = "0.1.0"
__all__ = ["Document", "Evidence", "ValidationResult", "RelevanceValidator", "validate", "DEFAULT_WEIGHTS"]
