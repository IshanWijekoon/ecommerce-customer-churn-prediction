"""Data loading and validation."""

from src.data.loader import load_data_dict, load_ecommerce_data, load_raw_workbook
from src.data.validation import ValidationResult, validate_ecommerce_frame

__all__ = [
    "load_data_dict",
    "load_ecommerce_data",
    "load_raw_workbook",
    "ValidationResult",
    "validate_ecommerce_frame",
]
