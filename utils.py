import numpy as np
import pandas as pd
import random
import math
import os
import torch
from typing import List, Tuple, Optional, Union
import statistics
import argparse
import json
import pickle
from sklearn.utils import shuffle
from typing import Optional
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]


def two_sided_olshen(cloud, coverage, B=10):
    cloud, fix = degenerate_fix_factory(cloud)
    bootstraps = np.random.choice(np.arange(cloud.shape[0]), size=(B, cloud.shape[0]))
    bootstraps = torch.tensor(bootstraps)
    clouds = cloud[bootstraps]
    maxes = torch.empty((B, cloud.shape[0]))
    for i, cloud_b in enumerate(clouds):
        zscores_ = torch.empty_like(cloud)
        for j, col in enumerate(cloud_b.T):
            median = col.median()
            above_mask = col >= median
            below_mask = col <= median
            above = col[above_mask]
            below = col[below_mask]
            zscores_[above_mask, j] = (above - median) / ((median - above).square().sum() / (above.shape[0] - 1)).sqrt()
            zscores_[below_mask, j] = (below - median) / ((median - below).square().sum() / (below.shape[0] - 1)).sqrt()
        maxes[i] = zscores_.nan_to_num().abs().max(dim=-1)[0]
    median_cloud = torch.median(cloud, dim=0)[0]
    sigma_minus = torch.empty_like(median_cloud)
    sigma_plus = torch.empty_like(median_cloud)
    for i, col in enumerate(cloud.T):
        median = median_cloud[i]
        above = col[col > median]
        below = col[col < median]
        sigma_plus[i] = ((above - median).square().sum() / (above.shape[0] - 1)).sqrt()
        sigma_minus[i] = ((below - median).square().sum() / (below.shape[0] - 1)).sqrt()

    def helper(p):
        k = torch.quantile(maxes, q=p, interpolation='higher')
        upper_bounds = median_cloud + k * sigma_plus
        lower_bounds = median_cloud - k * sigma_minus
        orthotope = fix(torch.stack([lower_bounds, upper_bounds]))
        orthotope = surv_orthotope(orthotope)
        return orthotope

    if isinstance(coverage, float):
        return helper(coverage)
    elif isinstance(coverage, list):
        return [helper(p) for p in coverage]
    elif isinstance(coverage, dict):
        return {p: helper(p) for p in coverage}
    elif isinstance(coverage, np.ndarray):
        return np.array([helper(p) for p in coverage])
    elif isinstance(coverage, torch.Tensor):
        return torch.stack([helper(p) for p in coverage])
    else:
        raise TypeError(f'coverage must be float, list, dict, or np.ndarray, not {type(coverage)}')


def degenerate_fix_factory(cloud, epsilon=1e-6):
    degenerate = torch.isclose(cloud.max(dim=0)[0], cloud.min(dim=0)[0])
    deg_cloud = cloud[:, degenerate]
    nondeg_cloud = cloud[:, ~degenerate]
    mean_deg = deg_cloud.mean(dim=0)
    lower_bounds = mean_deg - epsilon
    upper_bounds = mean_deg + epsilon

    def fix(nondeg_orthotope):
        fixed_orthotope = torch.empty((2, cloud.shape[1]))
        fixed_orthotope[:, degenerate] = torch.stack([lower_bounds, upper_bounds])
        fixed_orthotope[:, ~degenerate] = nondeg_orthotope
        return fixed_orthotope

    return nondeg_cloud, fix


def surv_orthotope(orthotope):
    orthotope = torch.clamp(orthotope, 0, 1)
    inv_idx = torch.arange(orthotope.shape[1] - 1, -1, -1).long()
    orthotope[0] = torch.cummax(orthotope[0, inv_idx], dim=0)[0][inv_idx]
    orthotope[1] = torch.cummin(orthotope[1], dim=0)[0]
    return orthotope


def count_parameters(model):
    res = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
            res += p.numel()
            # print(p)
    return res


def save_params(
        config: argparse.Namespace
) -> str:
    """
    Saves args for reproducing results
    """
    dir_ = os.getcwd()
    path = f"{dir_}/runs/{config.dataset}/{config.model}" \
           f"/{config.timestamp}"

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/commandline_args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    return path


def save_predictions(
        path: str,
        exp_num: int,
        model,
        data_test: pd.DataFrame
) -> None:
    """
    Saves model, and test set
    """
    subpath = os.path.join(path, f'split_{exp_num}')
    if not os.path.exists(subpath):
        os.makedirs(subpath)

    torch.save(model.state_dict(), f"{subpath}/model.pt")
    data_test.to_pickle(f"{subpath}/testset.pkl")


def print_performance(
        con: list = None,
        ibs: list = None,
        l1_unc: list = None,
        l1_hinge: list = None,
        l1_margin: list = None,
        pvalues: list = None,
        cvge: list = None,
        thk: list = None,
        path: str = None
) -> None:
    """
    Print performance using mean and std. And also save to files.
    """
    prf = f""
    prf += f"Concordance mean: {statistics.mean(con)}; " \
           f"standard deviation: {statistics.stdev(con)}\n" if con is not None else f""
    prf += f"IBS mean: {statistics.mean(ibs)}; " \
           f"standard deviation: {statistics.stdev(ibs)}\n" if ibs is not None else f""
    prf += f"L1-uncensored mean: {statistics.mean(l1_unc)}; " \
           f"standard deviation: {statistics.stdev(l1_unc)}\n" if l1_unc is not None else f""
    prf += f"L1-hinge mean: {statistics.mean(l1_hinge)}; " \
           f"standard deviation: {statistics.stdev(l1_hinge)}\n" if l1_hinge is not None else f""
    prf += f"L1-margin mean: {statistics.mean(l1_margin)}; " \
           f"standard deviation: {statistics.stdev(l1_margin)}\n" if l1_margin is not None else f""
    prf += f"D-Calibration: model calibrated {sum(i >= 0.05 for i in pvalues)} " \
           f"out of {len(pvalues)} times\n" if pvalues is not None else f""
    if cvge is not None:
        for i in range(len(cvge)):
            prf += f"Coverage run#{i}: {cvge[i]}\n"
    prf += f"Thickness mean: {statistics.mean(thk)}; " \
           f"standard deviation: {statistics.stdev(thk)}\n" if thk is not None else f""
    print(prf)

    if path is not None:
        prf_dict = {
            'con': con,
            'ibs': ibs,
            'l1_unc': l1_unc,
            'l1_hinge': l1_hinge,
            'l1_margin': l1_margin,
            'pvalues': pvalues,
            'coverage': cvge,
            'thickness': thk
        }
        with open(f"{path}/performance.pkl", 'wb') as f:
            pickle.dump(prf_dict, f)

        with open(f"{path}/performance.txt", 'w') as f:
            f.write(prf)


def is_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    return (all(array[i] <= array[i + 1] for i in range(len(array) - 1)) or
            all(array[i] >= array[i + 1] for i in range(len(array) - 1)))


def make_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array


def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored


def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike
) -> (torch.Tensor, torch.Tensor):
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None
) -> torch.Tensor:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins


def train_val_test_stratified_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_val: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    assert frac_train >= 0 and frac_val >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_val + frac_test
    frac_train = frac_train / frac_sum
    frac_val = frac_val / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_val == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_val + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test


def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test


def stratified_folds_survival(
        dataset: pd.DataFrame,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        number_folds: int = 5
) -> list:
    event_times, event_indicators = event_times.tolist(), event_indicators.tolist()
    assert len(event_indicators) == len(event_times)

    indicators_and_times = list(zip(event_indicators, event_times))
    sorted_idx = [i[0] for i in sorted(enumerate(indicators_and_times), key=lambda v: (v[1][0], v[1][1]))]

    folds = [[sorted_idx[0]], [sorted_idx[1]], [sorted_idx[2]], [sorted_idx[3]], [sorted_idx[4]]]
    for i in range(5, len(sorted_idx)):
        fold_number = i % number_folds
        folds[fold_number].append(sorted_idx[i])

    training_sets = [dataset.drop(folds[i], axis=0) for i in range(number_folds)]
    testing_sets = [dataset.iloc[folds[i], :] for i in range(number_folds)]

    cross_validation_set = list(zip(training_sets, testing_sets))
    return cross_validation_set
