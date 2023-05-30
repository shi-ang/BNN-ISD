import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from scipy.interpolate import interp1d
from typing import Union
import argparse
from utils import compute_unique_counts, make_monotonic

from loss import mtlr_nll, cox_nll
from base_layers import BayesianLinear, BayesianElementwiseLinear, BayesianHorseshoeLayer


class BayesianBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

    @abstractmethod
    def log_variational_posterior(self):
        pass

    def get_name(self):
        return self._get_name()


class BayesianNetwork(BayesianBaseModel):
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.n_classes = 10
        self.l1 = BayesianLinear(28 * 28, 400, config)
        self.l2 = BayesianLinear(400, 400, config)
        self.l3 = BayesianLinear(400, self.n_classes, config)

    def forward(self, x, sample=False, n_samples=1):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample=sample, n_samples=n_samples))
        x = F.relu(self.l2(x, sample=sample, n_samples=n_samples))
        x = F.log_softmax(self.l3(x, sample=sample, n_samples=n_samples), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior\
               + self.l3.log_variational_posterior

    def sample_elbo(self, x, target, dataset_size):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        outputs = outputs.reshape(n_samples, self.config.batch_size, self.n_classes)    # Checkpoint needed
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        nll = F.nll_loss(outputs.mean(dim=0), target, size_average=False)

        # Shouldn't here be batch size instead?
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.__init__(self.config)
        return self


class mtlr(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")

    def get_name(self):
        return self._get_name()


class BayesLinMtlr(BayesianBaseModel):
    """Multi-task logistic regression for individualised
    survival prediction with Bayesian layers.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.
    """

    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        config
            Configuration/hyper-parameters of the network.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianLinear(self.in_features, self.num_time_bins - 1, config)
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        outputs = self.l1(x, sample, n_samples)
        this_batch_size = x.shape[0]    # because the last batch may not be a complete batch.
        outputs = outputs.reshape(n_samples, this_batch_size, self.num_time_bins - 1)    # this can be deleted

        # forward only returns (w * x + b) for computing nll loss
        # survival curves will be generated using mtlr_survival() function.
        # return outputs
        G_with_samples = self.G.expand(n_samples, -1, -1)

        # b: n_samples; i: n_data; j: n_bin - 1; k: n_bin
        return torch.einsum('bij,bjk->bik', outputs, G_with_samples)

    def log_prior(self):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        return self.l1.log_prior

    def log_variational_posterior(self):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, y, dataset_size) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the evidence lower bound for a batch with sampling.
        :param x:
        :param y: Label
        :param dataset_size:
        :return:
        """
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = mtlr_nll(outputs.mean(dim=0), y, model=self, C1=0, average=False)
        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")


class BayesHsLinMtlr(BayesLinMtlr):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        super(BayesLinMtlr, self).__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianHorseshoeLayer(self.in_features, self.num_time_bins - 1, config)
        self.register_buffer("G",
                             torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins, requires_grad=True)))

    def fixed_point_update(self):
        """Calculates the update of the model parameters with fixed point updates equations"""
        return self.l1.fixed_point_update()


class BayesEleMtlr(BayesianBaseModel):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, self.num_time_bins - 1, config)
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        this_batch_size = x.shape[0]    # because the last batch may not be a complete batch.
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)
        outputs = outputs.reshape(n_samples, this_batch_size, self.num_time_bins - 1)    # this can be deleted, just for the safety

        # forward only returns (w * x + b) for computing nll loss
        # survival curves will be generated using mtlr_survival() function.
        # return outputs
        G_with_samples = self.G.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: n_bin - 1; k: n_bin
        return torch.einsum('bij,bjk->bik', outputs, G_with_samples)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(
            self,
            x,
            y,
            dataset_size
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = mtlr_nll(outputs.mean(dim=0), y, model=self, C1=0, average=False)
        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"hidden_size={self.hidden_size}), "
                f"num_time_bins={self.num_time_bins})")


class BayesMtlr(BayesEleMtlr):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        config
            Configuration/hyper-parameters of the network.
        """
        super(BayesEleMtlr, self).__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, self.num_time_bins - 1, config)
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))


class BayesHsMtlr(BayesEleMtlr):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        super(BayesEleMtlr, self).__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianHorseshoeLayer(self.in_features, self.config.hidden_size, config)
        self.l2 = BayesianLinear(self.config.hidden_size, self.num_time_bins - 1, config)
        self.register_buffer("G",
                             torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins, requires_grad=True)))

    def fixed_point_update(self):
        """Calculates the update of the model parameters with fixed point update update equations."""
        return self.l1.fixed_point_update()


class CoxPH(nn.Module):
    """Cox proportional hazard model for individualised survival prediction."""

    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = nn.Linear(self.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.l1(x)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()


class BayesLinCox(BayesianBaseModel):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None

        self.l1 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        outputs = self.l1(x, sample, n_samples)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, t, e, dataset_size) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)

        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / (dataset_size) + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"


class BayesHsLinCox(BayesLinCox):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super(BayesLinCox, self).__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianHorseshoeLayer(self.in_features, 1, config)

    def fixed_point_update(self):
        """Calculates the update of the model parameters with fixed point updates equations"""
        return self.l1.fixed_point_update()


class BayesEleCox(BayesianBaseModel):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)

        outputs = outputs.squeeze(dim=-1)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(
            self,
            x,
            t: torch.Tensor,
            e: torch.Tensor,
            dataset_size: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)

        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / (32 * dataset_size) + nll
        return loss, log_prior, log_variational_posterior / dataset_size, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"hidden_size={self.hidden_size})")


class BayesCox(BayesEleCox):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super(BayesEleCox, self).__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 1, config)


class BayesHsCox(BayesEleCox):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super(BayesEleCox, self).__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianHorseshoeLayer(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 1, config)

    def fixed_point_update(self):
        """Calculates the update of the model parameters with fixed point update update equations."""
        return self.l1.fixed_point_update()


def cox_survival(
        baseline_survival: torch.Tensor,
        linear_predictor: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the individual survival distributions based on the baseline survival curves and the liner prediction values.
    :param baseline_survival: (n_time_bins, )
    :param linear_predictor: (n_samples, n_data)
    :return:
    The invidual survival distributions. shape = (n_samples, n_time_bins)
    """
    n_sample = linear_predictor.shape[0]
    n_data = linear_predictor.shape[1]
    risk_score = torch.exp(linear_predictor)
    survival_curves = torch.empty((n_sample, n_data, baseline_survival.shape[0]), dtype=torch.float).to(linear_predictor.device)
    for i in range(n_sample):
        for j in range(n_data):
            survival_curves[i, j, :] = torch.pow(baseline_survival, risk_score[i, j])
    return survival_curves


def baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival


def mtlr_survival(
        logits: torch.Tensor,
        with_sample: bool = True
) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module)
        with size (n_samples, n_data, n_bins) or (n_data, n_bins).

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)


def mtlr_survival_at_times(
        logits: torch.Tensor,
        train_times: Union[torch.Tensor, np.ndarray],
        pred_times: np.ndarray
) -> np.ndarray:
    """Generates predicted survival curves at arbitrary timepoints using linear
    interpolation.

    Notes
    -----
    This function uses scipy.interpolate internally and returns a Numpy array,
    in contrast with `mtlr_survival`.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    train_times
        Time bins used for model training. Must have the same length as the
        first dimension of `pred`.
    pred_times
        Array of times used to compute the survival curve.

    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values
        are linearly interpolated at timepoints not used for training.
    """
    train_times = np.pad(train_times, (1, 0))
    surv = mtlr_survival(logits).detach().cpu().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(
        logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits: torch.Tensor) -> torch.Tensor:
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)
