import math
import numpy as np
import argparse
import torch
import torch.nn as nn

from distributions import ParametrizedGaussian, ScaleMixtureGaussian, InverseGamma


class BayesianHorseshoeLayer(nn.Module):
    """
    Single linear layer of a horseshoe prior.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            config: argparse.Namespace
    ):
        """
        Initialize HS layer using non-centered parameterization and variational approximation.

        :param in_features: number of input features
        :param out_features: number of output features
        :param config: hyperparameters, instance of class argparse.Namespace
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Initialization of parameters of variational distribution
        # weight parameters
        self.beta_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]) * config.rho_scale)
        self.beta = ParametrizedGaussian(self.beta_mu, self.beta_rho)
        # bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

        # Initialization of parameters of prior distribution
        # weight parameters
        self.prior_tau_shape = torch.Tensor([0.5]).to(config.device)

        # local shrinkage parameters
        self.prior_lambda_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_lambda_rate = torch.Tensor([1 / config.weight_cauchy_scale ** 2]).to(config.device)

        # local shrinkage parameters
        self.lambda_shape = self.prior_lambda_shape * torch.ones(in_features).to(config.device)
        self.lambda_rate = self.prior_lambda_rate * torch.ones(in_features).to(config.device)
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        # Sample from half-Cauchy to initialize the mean of log_tau
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution over tau
        distr = torch.distributions.HalfCauchy(1 / torch.sqrt(self.prior_lambda_rate))
        sample = distr.sample(torch.Size([in_features])).squeeze()
        self.log_tau_mean = nn.Parameter(torch.log(sample))
        self.log_tau_rho = nn.Parameter(torch.ones(in_features) * config.rho_scale)
        self.log_tau = ParametrizedGaussian(self.log_tau_mean, self.log_tau_rho)

        # global shrinkage parameters
        self.prior_v_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_theta_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_theta_rate = torch.Tensor([1 / config.global_cauchy_scale ** 2]).to(config.device)

        # global shrinkage parameters
        self.theta_shape = self.prior_theta_shape
        self.theta_rate = self.prior_theta_rate
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        # Sample from half-Cauchy to initialize the mean of log_v
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution over v
        distr = torch.distributions.HalfCauchy(1 / torch.sqrt(self.prior_theta_rate))
        sample = distr.sample()
        self.log_v_mean = nn.Parameter(torch.log(sample))
        self.log_v_rho = nn.Parameter(torch.ones(1) * config.rho_scale)
        self.log_v = ParametrizedGaussian(self.log_v_mean, self.log_v_rho)

    @property
    def log_prior(self):
        """
        Computes the expectation of the log of the prior p under the variational posterior q
        """

        # Calculate E_q[ln p(\tau | \lambda)] + E[ln p(\lambda)]
        # E_q[ln p(\tau | \lambda)] for the weights
        shape = self.prior_tau_shape
        exp_lambda_inverse = self.lambda_.exp_inverse()
        exp_log_lambda = self.lambda_.exp_log()
        exp_log_tau = self.log_tau.mu
        exp_tau_inverse = torch.exp(- self.log_tau.mu + 0.5 * self.log_tau.sigma ** 2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda,
                                                      exp_log_tau, exp_tau_inverse)

        # E_q[ln p(\lambda)] for the weights
        shape = self.prior_lambda_shape
        rate = self.prior_lambda_rate
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, torch.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)

        # E_q[ln p(v | \theta)] for the global shrinkage parameter
        shape = self.prior_v_shape
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        exp_log_v = self.log_v.mu
        exp_v_inverse = torch.exp(- self.log_v.mu + 0.5 * self.log_v.sigma ** 2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta,
                                                      exp_log_v, exp_v_inverse)

        # E_q[ln p(\theta)] for the global shrinkage parameter
        shape = self.prior_theta_shape
        rate = self.prior_theta_rate
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, torch.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global

        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mu, self.beta.sigma) + exp_log_gaussian(self.bias.mu, self.bias.sigma)

        return log_gaussian + log_inv_gammas

    @property
    def log_variational_posterior(self):
        """
        Computes the log of the variational posterior by computing the entropy.

        The entropy is defined as - sum[q(theta) * log(q(theta))]. The log of the
        variational posterior is given by sum[q(theta) * log(q(theta))] = -E[-log(q(theta))] = H[q(theta)].
        Therefore, we compute the entropy and return -entropy.

        Tau and v follow log-Normal distributions. The entropy of a log normal
        is the entropy of the normal distribution + the mean.
        """
        entropy = (self.beta.entropy() +
                   self.log_tau.entropy() + torch.sum(self.log_tau.mu) +
                   self.lambda_.entropy() + self.bias.entropy() +
                   self.log_v.entropy() + torch.sum(self.log_v.mu) +
                   self.theta.entropy())

        if torch.isnan(entropy).item():
            print('self.beta.entropy(): ', self.beta.entropy())
            print('beta mean: ', self.beta.mu)
            print('beta std: ', self.beta.sigma)
            raise Exception("entropy/log_variational_posterior computation ran into nan!")

        return -entropy

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        """
        Performs a forward pass through the layer, that is, computes the layer output for a given input batch.

        Args:
            x: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """
        if self.training or sample:
            beta = self.beta.sample(n_samples)
            log_tau = torch.unsqueeze(self.log_tau.sample(n_samples), 1)
            log_v = torch.unsqueeze(self.log_v.sample(n_samples), 1)
            bias = self.bias.sample(n_samples)
        else:
            print("No sampling")
            beta = self.beta.mu.expand(n_samples, -1, -1)
            log_tau = self.log_tau.mu.expand(n_samples, -1, -1)
            log_v = self.log_v.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        weight = beta * log_tau * log_v
        x = x.expand(n_samples, -1, -1)

        result = torch.einsum('bij,bkj->bik', x, weight) + bias
        return result

    def fixed_point_update(self):
        """
        Calculates fixed point updates of lambda_ and theta

        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        and also this JMLR paper: https://jmlr.org/papers/v20/19-236.html
        """
        new_shape = torch.Tensor([1]).to(self.config.device)
        # new lambda rate is given by E[1/tau_i] + 1/b_0^2
        new_lambda_rate = (torch.exp(- self.log_tau.mu + 0.5 * (self.log_tau.sigma ** 2)) +
                           self.prior_lambda_rate).to(self.config.device)

        # new theta rate is given by E[1/v] + 1/b_g^2
        new_theta_rate = (torch.exp(- self.log_v.mu + 0.5 * (self.log_v.sigma ** 2)) +
                          self.prior_theta_rate).to(self.config.device)

        self.lambda_.update(new_shape, new_lambda_rate)
        self.theta.update(new_shape, new_theta_rate)

    def reset_parameters(self):
        """Reset parameter for tau, v, and beta.

        We don't need to reset lambda and theta because we will use fixed point updates to update them.
        """
        # Reinitialization of parameters of variational distribution
        # weight parameters
        nn.init.xavier_uniform_(self.beta_mu)
        nn.init.constant_(self.beta_rho, self.config.rho_scale)
        self.beta = ParametrizedGaussian(self.beta_mu, self.beta_rho)
        # bias parameters
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

        # Sample from half-Cauchy to reinitialize the mean of log_tau
        distr = torch.distributions.HalfCauchy(1 / torch.sqrt(self.prior_lambda_rate))
        sample = distr.sample(torch.Size([self.in_features])).squeeze()
        # self.log_tau_mean = nn.Parameter(torch.log(sample))
        for i in range(len(sample)):
            nn.init.constant_(self.log_tau_mean[i], sample[i])
        nn.init.constant_(self.log_tau_rho, self.config.rho_scale)
        self.log_tau = ParametrizedGaussian(self.log_tau_mean, self.log_tau_rho)

        # Sample from half-Cauchy to reinitialize the mean of log_v
        distr = torch.distributions.HalfCauchy(1 / torch.sqrt(self.prior_theta_rate))
        sample = distr.sample().squeeze()
        # self.log_v_mean = nn.Parameter(torch.log(sample))
        nn.init.constant_(self.log_v_mean, sample)
        nn.init.constant_(self.log_v_rho, self.config.rho_scale)
        self.log_v = ParametrizedGaussian(self.log_v_mean, self.log_v_rho)

        # Reset the parameters for fix point updates.
        self.lambda_.update(self.lambda_shape, self.lambda_rate)
        self.theta.update(self.theta_shape, self.theta_rate)


class BayesianElementwiseLinear(nn.Module):
    """
    Single elementwise linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            input_output_size: int,
            config: argparse.Namespace
    ):
        """
        Initialize gaussian layer using reparameterization.

        :param input_output_size: number of input features
        :param config: hyperparameters
        """
        super().__init__()
        self.input_output_size = input_output_size
        self.config = config
        if self.config.mu_scale is None:
            self.config.mu_scale = 1. * np.sqrt(6. / input_output_size)

        self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(input_output_size)),
                                          -self.config.mu_scale, self.config.mu_scale)
        self.weight_rho = nn.Parameter(torch.ones([input_output_size]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)

        self.weight_prior = ScaleMixtureGaussian(config.pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # b: n_samples; i: n_data; j: input output size; k: input output size
        weight = torch.einsum('bj, jk->bjk', weight,
                              torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device))
        x = x.expand(n_samples, -1, -1)
        return torch.einsum('bij,bjk->bik', x, weight)

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.uniform_(self.weight_mu, -self.config.mu_scale, self.config.mu_scale)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)


class BayesianLinear(nn.Module):
    """
    Single linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            config: argparse.Namespace,
            use_mixture: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights
        self.config = config
        if self.config.mu_scale is None:
            self.weight_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        else:
            self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(out_features, in_features)),
                                              -self.config.mu_scale, self.config.mu_scale)

        self.weight_rho = nn.Parameter(torch.ones([out_features, in_features]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * self.config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        if use_mixture:
            pi = config.pi
        else:
            pi = 1
        self.weight_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)
        self.bias_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
            bias = self.bias.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # For a single layer network, x would have 2 dimension [n_data, n_feature]
        # But sometime x would be the sampled output from the previous layer,
        # which will have 3 dimension [n_samples, n_data, n_feature]
        n_data = x.shape[-2]
        bias = bias.repeat(1, n_data, 1)
        # If x is 3-d, this expand command will make x remains the same.
        x = x.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: input features size; k: output size
        return torch.einsum('bij,bkj->bik', x, weight) + bias

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)


def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
    """
    Calculates the expectation of the log of an inverse gamma distribution p under
    the posterior distribution q
    E_q[log p(x | shape, rate)]


    Args:
    shape: float, the shape parameter of the gamma distribution
    exp_rate: torch tensor, the expectation of the rate parameter under q
    exp_log_rate: torch tensor, the expectation of the log of the rate parameter under q
    exp_log_x: torch tensor, the expectation of the log of the random variable under q
    exp_x_inverse: torch tensor, the expectation of the inverse of the random variable under q

    Returns:
    exp_log: torch tensor, E_q[log p(x | shape, rate)]
    """
    exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x - exp_rate * exp_x_inverse

    # We need to sum over all components since this is a vectorized implementation.
    # That is, we compute the sum over the individual expected values. For example,
    # in the horseshoe BLR model we have one local shrinkage parameter for each weight
    # and therefore one expected value for each of these shrinkage parameters.
    return torch.sum(exp_log)


def exp_log_gaussian(mean, std):
    """
    Calculates the expectation of the log of a Gaussian distribution p under the posterior distribution q
    E_q[log p(x)] - see note log_prior_gaussian.pdf

    Args:
    mean: torch tensor, the mean of the posterior distribution
    std: torch tensor, the standard deviation of the posterior distribution

    Returns:
    exp_gaus: torch tensor, E_q[p(x)]


    Comment about how this function is vectorized:
    Every component beta_i follows a univariate Gaussian distribution, and therefore has
    a scalar mean and a scalar variance. We can combine all components of beta into a
    diagonal Gaussian distribution, which has a mean vector of the same length as the
    beta vector, and a standard deviation vector of the same length. By summing over the
    mean vector and over the standard deviations, we therefore sum over all components of beta.
    """
    dim = mean.shape[0] * mean.shape[1]
    exp_gaus = - 0.5 * dim * (torch.log(torch.tensor(2 * math.pi))) - 0.5 * (torch.sum(mean ** 2) + torch.sum(std ** 2))
    return exp_gaus
