import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_curve_with_bar(
        time_bins: torch.Tensor,
        mean_outputs: torch.Tensor,
        upper_outputs: torch.Tensor,
        lower_outputs: torch.Tensor,
        index: int = 0
) -> None:
    plt.rcParams["figure.figsize"] = [4, 3]
    plt.plot(time_bins.cpu().numpy(), mean_outputs.cpu().numpy()[index, :], '-')
    plt.fill_between(time_bins.cpu().numpy(), upper_outputs.cpu().numpy()[index, :],
                     lower_outputs.cpu().numpy()[index, :], color='gray', alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig("/home/shiang/Desktop/survival_with_credible_region.png", dpi=200)


def plot_weights_dist(
        means: np.ndarray,
        variances: np.ndarray,
        feature_names: list,
        path: str = None
) -> None:
    # plt.rcParams["figure.figsize"] = [12, 9]
    # plt.rcParams["figure.autolayout"] = True
    # plt.rcParams["figure.dpi"] = 400
    # lower_bound = min(means - 3 * variances)
    # upper_bound = max(means + 3 * variances)
    # x_range = np.linspace(lower_bound, upper_bound, 500)
    # for i in range(len(means)):
    #     plt.plot(x_range, stats.norm.pdf(x_range, means[i], variances[i]), label=feature_names[i])
    #
    # plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    # if y_lims is None:
    #     y_lims = [0, 2.5]
    # plt.ylim(y_lims)
    #
    # plt.xlabel("Values")
    # plt.title("Probability Distribution for Each Weight")
    # plt.show()

    # plt.rcParams["figure.figsize"] = [3.2, 2] # 6.4 for synthetic-i, 2 for ii
    y_pos = np.arange(len(feature_names))
    fig, ax = plt.subplots()
    ax.barh(y_pos, np.abs(means), xerr=variances)
    ax.set_xlabel('|Weight|')
    # if y_lims is None:
    #     y_lims = [0, 2.5]
    # plt.ylim(y_lims)
    # plt.ylim([-0.7, 9.7]) # 49.7 for synthetic-i, 9.7 for ii
    plt.yticks(y_pos, feature_names)
    # plt.title("Probability Distribution for Each Weight")
    plt.tight_layout()
    if path is not None:
        plt.savefig(f"{path}.png", dpi=300)
    else:
        plt.show()


def plot_weights_hist(
        means: np.ndarray,
) -> None:
    log_means = np.log10(np.abs(means))
    bins = 20
    fig, ax = plt.subplots()
    ax.hist(log_means, bins)
    ax.set_xlabel('log(|Weight|)')
    ax.set_ylabel('Counts')
    plt.title("Weights Histogram")
    plt.tight_layout()
    plt.show()
