from dataclasses import dataclass
import numpy as np
from .core import mode


# TODO use precomputed groups here for performance
def evaluate_sample_pruned(sample, utility, group, k, threshold, group_weights):

    predictions_with_groups = zip(sample['retrieved_answers'], sample['retrieved_websites'])

    pruned_predictions = [prediction for prediction, website in predictions_with_groups
                          if group(website) not in group_weights or group_weights[group(website)] >= threshold]

    topk = pruned_predictions[:k]

    if len(topk) > 0:
        prediction = mode(topk)
        return utility(sample, prediction)
    else:
        return 0.0


def evaluate_pruned(samples, utility, group, k, threshold, group_weights, normalize=False):
    aggregate_utility = 0.0

    for sample in samples:
        aggregate_utility += evaluate_sample_pruned(sample, utility, group, k, threshold, group_weights)

    if normalize:
        aggregate_utility /= len(samples)

    return aggregate_utility


@dataclass
class TuningResult:
    achieved_utilities: list[float]
    best_utility: float
    best_threshold: float
    best_percentile: int


def tune_pruning_threshold(samples, group_weights, percentile_range, utility, group, k, normalize=False):

    best_utility = 0.0
    best_threshold = 0.0
    best_percentile = 0

    achieved_utilities = []

    for percentile in percentile_range:
        threshold = np.percentile(list(group_weights.values()), percentile)

        achieved_utility = evaluate_pruned(samples, utility, group, k, threshold, group_weights, normalize=normalize)

        achieved_utilities.append(achieved_utility)

        if achieved_utility >= best_utility:
            best_utility = achieved_utility
            best_threshold = threshold
            best_percentile = percentile

    return TuningResult(achieved_utilities, best_utility, best_threshold, best_percentile)
