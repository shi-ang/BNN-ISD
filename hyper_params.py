import argparse
import json
import math


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="SUPPORT",
                        choices=["Synthetic-I", "Synthetic-II", "Synthetic-III",
                                 "SUPPORT", "NACD", "MIMIC"],
                        help="Dataset name.")
    parser.add_argument('--model', type=str, default="BayesianMTLR",
                        choices=["MTLR", "BayesianHorseshoeLinearMTLR",
                                 "BayesianElementwiseMTLR", "BayesianHorseshoeMTLR",
                                 "BayesianLinearMTLR", "BayesianMTLR",
                                 "CoxPH", "BayesianHorseshoeLinearCox",
                                 "BayesianElementwiseCox", "BayesianHorseshoeCox",
                                 "BayesianLinearCox", "BayesianCox"],
                        help="Model name.")

    # General parameters
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help="Number of maximum training epoch.")
    parser.add_argument('--patience', type=int, default=50,
                        help="Number of patience epoch for convergence. Only used if 'early_stop' == True")
    parser.add_argument('--early_stop', type=bool, default=True,
                        help="Whether using early stop for training.")
    parser.add_argument('--seed', type=int, default=39,
                        help="Random seed for initialization")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.00008,  # 0.005 for cox, 0.00008 for mtlr
                        help="Learning rate.")
    parser.add_argument('--verbose', type=bool, default=True,
                        help="Verbose.")
    parser.add_argument('--hidden_size', type=int, default=50,
                        help="Hidden neurons in 2-layer network.")
    parser.add_argument('--dropout', type=float, default=0.6,
                        help="Dropout rate.")

    # BNN parameters
    parser.add_argument('--mu_scale', type=float, default=None,
                        help="Scaling rate for initialize the mean of weights.")
    parser.add_argument('--rho_scale', type=float, default=-5.,
                        help="Scaling rate for initialize the standard deviation of weights.")
    parser.add_argument('--n_samples_train', type=int, default=10,
                        help="Number of samples to draw from the variational posterior for calculating ELBO.")
    parser.add_argument('--n_samples_test', type=int, default=100,
                        help="Number of samples to draw for predicting.")

    # Mixture Gaussian parameters
    parser.add_argument('--pi', type=float, default=0.5,
                        help="Mixing coefficient for mixture Gaussian Distribution.")
    parser.add_argument('--sigma1', type=float, default=1,
                        help="Standard deviation 1 for mixture Gaussian Distribution.")
    parser.add_argument('--sigma2', type=float, default=math.exp(-6),
                        help="Standard deviation 2 for mixture Gaussian Distribution.")

    # Horseshoe parameters
    parser.add_argument('--weight_cauchy_scale', type=float, default=1,
                        help="Half-cauchy scale for local shrinkage parameters lambda.")
    parser.add_argument('--global_cauchy_scale', type=float, default=1,
                        help="Half-cauchy scale for global shrinkage parameter theta.")

    # MTLR parameters
    parser.add_argument('--c1', type=float, default=0.01,
                        help="Hyperparameter for the penalty term. Not used for BNN-based model")

    args = parser.parse_args()
    return args


def load_parser(file: str):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(file, 'r') as f:
        args.__dict__ = json.load(f)
    return args
