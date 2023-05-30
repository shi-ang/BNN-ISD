import torch
import numpy as np

from SurvivalEVAL.Evaluations.util import check_and_convert
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike


CiStatistics = {'80%': 1.282,
                '85%': 1.440,
                '90%': 1.645,
                '95%': 1.960}


def thickness(
        time_bins: torch.Tensor,
        upper: torch.Tensor,
        lower: torch.Tensor
) -> float:
    """
    :param time_bins: (n_bin, ), the time bins of the survival curves
    :param upper: (dataset_size, n_bin), the 95% upper bound for the ensemble survival curves of each patient
    :param lower: (dataset_size, n_bin), the 95% lower bound for the ensemble survival curves of each patient
    """
    area = torch.trapz(y=upper-lower, x=time_bins, dim=1).mean().item()
    max_area = time_bins.max().item()
    return area / max_area


def coverage(
        time_bins: NumericArrayLike,
        upper: NumericArrayLike,
        lower: NumericArrayLike,
        true_times: NumericArrayLike,
        true_indicator: NumericArrayLike
) -> float:
    time_bins = check_and_convert(time_bins)
    upper, lower = check_and_convert(upper, lower)
    true_times, true_indicator = check_and_convert(true_times, true_indicator)
    true_indicator = true_indicator.astype(bool)
    covered = 0
    upper_median_times = predict_median_survival_times(upper, time_bins, round_up=True)
    lower_median_times = predict_median_survival_times(lower, time_bins, round_up=False)
    covered += 2 * np.logical_and(upper_median_times[true_indicator] >= true_times[true_indicator],
                                  lower_median_times[true_indicator] <= true_times[true_indicator]).sum()
    covered += np.sum(upper_median_times[~true_indicator] >= true_times[~true_indicator])
    total = 2 * true_indicator.sum() + (~true_indicator).sum()
    return covered / total


def coverage_curves(
        upper: torch.Tensor,
        lower: torch.Tensor,
        test_curves: torch.Tensor
) -> float:
    upper = upper.cpu().detach().numpy()
    lower = lower.cpu().detach().numpy()
    test_curves = test_curves.cpu().detach().numpy()
    return ((upper >= test_curves) & (lower <= test_curves)).mean()


def predict_median_survival_times(
        survival_curves: np.ndarray,
        times_coordinate: np.ndarray,
        round_up: bool = True
):
    median_probability_times = np.zeros(survival_curves.shape[0])
    max_time = times_coordinate[-1]
    slopes = (1 - survival_curves[:, -1]) / (0 - max_time)

    if round_up:
        # Find the first index in each row that are smaller or equal than 0.5
        times_indices = np.where(survival_curves <= 0.5, survival_curves, -np.inf).argmax(axis=1)
    else:
        # Find the last index in each row that are larger or equal than 0.5
        times_indices = np.where(survival_curves >= 0.5, survival_curves, np.inf).argmin(axis=1)

    need_extend = survival_curves[:, -1] > 0.5
    median_probability_times[~need_extend] = times_coordinate[times_indices][~need_extend]
    median_probability_times[need_extend] = (max_time + (0.5 - survival_curves[:, -1]) / slopes)[need_extend]

    return median_probability_times
