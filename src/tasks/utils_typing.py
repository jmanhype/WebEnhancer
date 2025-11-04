"""Typing utilities for annotation parsing and validation."""
from typing import List, Any, Optional
import ast


class AnnotationList:
    """Wrapper for parsing and managing annotation lists from model output."""

    def __init__(self, annotations: List[Any]):
        """Initialize annotation list.

        Args:
            annotations: List of annotation objects
        """
        self.annotations = annotations

    @classmethod
    def from_output(cls, output_string: str, task_module: str = "guidelines") -> "AnnotationList":
        """Parse annotations from model output string.

        Args:
            output_string: Raw output string from the model containing annotations
            task_module: Name of the module containing annotation definitions

        Returns:
            AnnotationList instance with parsed annotations

        Raises:
            SyntaxError: If output string cannot be parsed as valid Python
            ValueError: If output format is invalid
        """
        try:
            # Try to parse the output as a Python literal
            parsed = ast.literal_eval(output_string.strip())

            if not isinstance(parsed, list):
                parsed = [parsed]

            return cls(parsed)

        except (SyntaxError, ValueError) as e:
            # If parsing fails, return empty annotation list
            print(f"Warning: Failed to parse annotations from output: {e}")
            return cls([])

    def __iter__(self):
        """Make the annotation list iterable."""
        return iter(self.annotations)

    def __len__(self):
        """Return number of annotations."""
        return len(self.annotations)

    def __repr__(self):
        """String representation of annotation list."""
        return f"AnnotationList({self.annotations})"
