"""Utility class."""

import numpy as np


def adjust_predictions(predictions: np.ndarray, scale_factor: float = 1.3) -> np.ndarray:
    """Adjust predictions by multiplying them with a scale factor.

    :param predictions: Array of predictions to be adjusted
    :param scale_factor: Factor to scale the predictions by
    :return: Adjusted predictions array
    """
    return [round(pred * scale_factor, 2) for pred in predictions]

def c_statistic_harrell(pred, labels) -> float:
    """
    Compute Harrell's C-statistic, a concordance index used to evaluate 
    the predictive accuracy of survival models or models where ranking matters.

    The metric checks how well the predicted values preserve the ordering 
    of true outcomes — higher predicted values should correspond to higher labels.

    Args:
        pred (array-like): Predicted scores or risk values.
        labels (array-like): Ground truth outcome values (e.g., survival times or severity scores).

    Returns:
        float: C-statistic value between 0 and 1 (higher is better).
            Returns NaN if no comparable pairs exist.
    """
    total = 0
    matches = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[j] > 0 and abs(labels[i]) > labels[j]:
                total += 1
                if pred[j] > pred[i]:
                    matches += 1
    return matches / total if total > 0 else float('nan')