"""Scoring utilities for evaluating relation extraction performance."""
from typing import List, Dict, Any, Type
from collections import defaultdict


class RelationScorer:
    """Base class for scoring relation extraction results."""

    valid_types: List[Type] = []

    def __call__(
        self,
        reference: List[List[Any]],
        predictions: List[List[Any]]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score.

        Args:
            reference: List of reference annotation lists (gold standard)
            predictions: List of predicted annotation lists

        Returns:
            Dictionary containing precision, recall, and f1 scores
        """
        if not reference or not predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Flatten lists
        ref_flat = [item for sublist in reference for item in sublist]
        pred_flat = [item for sublist in predictions for item in sublist]

        # Calculate metrics
        true_positives = len(set(str(x) for x in ref_flat) & set(str(x) for x in pred_flat))
        false_positives = len(pred_flat) - true_positives
        false_negatives = len(ref_flat) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
