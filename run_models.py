import math

import numpy as np
import argparse
import pandas as pd
from tqdm import trange
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from datasets import make_data
from loss import mtlr_nll, cox_nll
from models import BayesianBaseModel
from models import mtlr_survival, mtlr, BayesHsLinMtlr, BayesEleMtlr, BayesHsMtlr, BayesLinMtlr, BayesMtlr
from models import cox_survival, CoxPH, BayesHsLinCox, BayesEleCox, BayesHsCox, BayesLinCox, BayesCox
from utils import (save_predictions, save_params, train_val_test_stratified_split, reformat_survival, print_performance,
                   NumericArrayLike, make_time_bins, two_sided_olshen)
from plots import plot_weights_dist, plot_curve_with_bar, plot_weights_hist
from ci_evaluation import thickness, coverage
from hyper_params import load_parser, generate_parser

from SurvivalEVAL import BaseEvaluator

credible_region_sizes = np.arange(0.1, 1, 0.1)


def train_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        path: str,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    data_train, _, data_val = train_val_test_stratified_split(data_train, stratify_colname='both',
                                                              frac_train=0.9, frac_test=0.1,
                                                              random_state=random_state)

    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    if isinstance(model, mtlr):
        x, y = reformat_survival(data_train, time_bins)
        x_val, y_val = reformat_survival(data_val, time_bins)
        x_val, y_val = x_val.to(device), y_val.to(device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
        for i in pbar:
            nll_loss = 0
            for xi, yi in train_loader:
                xi, yi = xi.to(device), yi.to(device)
                optimizer.zero_grad()
                y_pred = model.forward(xi)
                loss = mtlr_nll(y_pred, yi, model, C1=config.c1, average=False)

                loss.backward()
                optimizer.step()

                nll_loss += (loss / train_size).item()
            logits_outputs = model.forward(x_val)
            eval_nll = mtlr_nll(logits_outputs, y_val, model, C1=0, average=True)
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"nll-loss = {nll_loss:.4f}; "
                                 f"Validation nll = {eval_nll.item():.4f};")
            if config.early_stop:
                if best_val_nll > eval_nll:
                    best_val_nll = eval_nll
                    best_ep = i
                    torch.save(model, path)
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break
    elif isinstance(model, BayesEleMtlr) or isinstance(model, BayesLinMtlr):
        x, y = reformat_survival(data_train, time_bins)
        x_val, y_val = reformat_survival(data_val, time_bins)
        x_val, y_val = x_val.to(device), y_val.to(device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
        for i in pbar:
            total_loss = 0
            total_log_likelihood = 0
            total_kl_divergence = 0
            for xi, yi in train_loader:
                xi, yi = xi.to(device), yi.to(device)
                optimizer.zero_grad()
                loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, yi, train_size)

                loss.backward()
                optimizer.step()

                if isinstance(model, BayesHsMtlr) or isinstance(model, BayesHsLinMtlr):
                    model.fixed_point_update()

                total_loss += loss.item() / train_size
                total_log_likelihood += log_likelihood.item() / train_size
                total_kl_divergence += (log_variational_posterior.item() -
                                        log_prior.item()) * config.batch_size / train_size**2

            val_loss, _, _, val_log_likelihood = model.sample_elbo(x_val, y_val, dataset_size=val_size)
            val_loss /= val_size
            val_log_likelihood /= val_size
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"Train: Total = {total_loss:.4f}, "
                                 f"KL = {total_kl_divergence:.4f}, "
                                 f"nll = {total_log_likelihood:.4f}; "
                                 f"Val: Total = {val_loss.item():.4f}, "
                                 f"nll = {val_log_likelihood.item():.4f}; ")
            if config.early_stop:
                if best_val_nll > val_loss:
                    best_val_nll = val_loss
                    best_ep = i
                    torch.save(model, path)
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break
    elif isinstance(model, CoxPH):
        x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                     torch.tensor(data_train["time"].values, dtype=torch.float),
                                     torch.tensor(data_train["event"].values, dtype=torch.float))
        x_val, t_val, e_val = (torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                               torch.tensor(data_val["time"].values, dtype=torch.float).to(device),
                               torch.tensor(data_val["event"].values, dtype=torch.float).to(device))

        train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
        model.config.batch_size = train_size

        for i in pbar:
            nll_loss = 0
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                optimizer.zero_grad()
                y_pred = model.forward(xi)
                nll_loss = cox_nll(y_pred, ti, ei, model, C1=config.c1)

                nll_loss.backward()
                optimizer.step()
                # here should have only one iteration
            logits_outputs = model.forward(x_val)
            eval_nll = cox_nll(logits_outputs, t_val, e_val, model, C1=0)
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"nll-loss = {nll_loss.item():.4f}; "
                                 f"Validation nll = {eval_nll.item():.4f};")
            if config.early_stop:
                if best_val_nll > eval_nll:
                    best_val_nll = eval_nll
                    best_ep = i
                    torch.save(model, path)
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break

    elif isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                     torch.tensor(data_train["time"].values, dtype=torch.float),
                                     torch.tensor(data_train["event"].values, dtype=torch.float))
        x_val, t_val, e_val = (torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                               torch.tensor(data_val["time"].values, dtype=torch.float).to(device),
                               torch.tensor(data_val["event"].values, dtype=torch.float).to(device))

        train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
        model.config.batch_size = train_size

        for i in pbar:
            total_loss = 0
            total_log_likelihood = 0
            total_kl_divergence = 0
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                optimizer.zero_grad()
                loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, ti, ei, train_size)

                loss.backward()
                optimizer.step()

                if isinstance(model, BayesHsLinCox) or isinstance(model, BayesHsCox):
                    model.fixed_point_update()

                total_loss += loss.item()
                total_log_likelihood += log_likelihood.item()
                total_kl_divergence += log_variational_posterior.item() - log_prior.item()

            val_loss, _, _, val_log_likelihood = model.sample_elbo(x_val, t_val, e_val, dataset_size=val_size)
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"Train: Total = {total_loss:.4f}, "
                                 f"KL = {total_kl_divergence:.4f}, "
                                 f"nll = {total_log_likelihood:.4f}; "
                                 f"Val: Total = {val_loss.item():.4f}, "
                                 f"nll = {val_log_likelihood.item():.4f}; ")
            if config.early_stop:
                if best_val_nll > val_loss:
                    best_val_nll = val_loss
                    best_ep = i
                    torch.save(model, path)
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break
    else:
        raise TypeError("Model type cannot be identified.")
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    model = torch.load(path)
    if isinstance(model, CoxPH) or isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model


def make_ensemble_mtlr_prediction(
        model: BayesianBaseModel,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()

    with torch.no_grad():
        # ensemble_output should have size: n_samples * dataset_size * n_bin
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = mtlr_survival(logits_outputs, with_sample=True)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = torch.cat([torch.tensor([0]), time_bins], 0).to(survival_outputs.device)
    return mean_survival_outputs, time_bins, survival_outputs


def make_ensemble_cox_prediction(
        model: BayesianBaseModel,
        x: torch.Tensor,
        config: argparse.Namespace
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = cox_survival(model.baseline_survival, logits_outputs)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = model.time_bins
    return mean_survival_outputs, time_bins, survival_outputs


def make_mtlr_prediction(
        model: mtlr,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = mtlr_survival(pred, with_sample=False)

    time_bins = torch.cat([torch.tensor([0]), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)


def make_cox_prediction(
        model: CoxPH,
        x: torch.Tensor,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = cox_survival(model.baseline_survival, pred)
        survival_curves = survival_curves.squeeze()

    time_bins = model.time_bins
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)


def main():
    args = generate_parser()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    torch.manual_seed(42)
    # torch.autograd.set_detect_anomaly(True)

    data, coef = make_data(args.dataset)
    assert "time" in data.columns and "event" in data.columns, "The event time variable and censor indicator " \
                                                               "variable is missing or need to be renamed."
    args.n_features = data.shape[1]
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    path = save_params(args)

    if 'true_time' in data.columns:
        feature_names = data.drop(["time", "event", "true_time"], axis=1).columns.to_list()
        num_features = args.n_features - 3
    else:
        feature_names = data.drop(["time", "event"], axis=1).columns.to_list()
        num_features = args.n_features - 2

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_train, _, data_test = train_val_test_stratified_split(data, stratify_colname='both',
                                                               frac_train=0.8, frac_test=0.2,
                                                               random_state=seed)
    if 'true_time' in data.columns:
        data_train = data_train.drop(columns=["true_time"])
        data_test = data_test.drop(columns=["time"])
        data_test = data_test.rename(columns={'true_time': 'time'})
        data_test.loc[:, "event"] = np.ones(data_test.shape[0])

    time_bins = make_time_bins(data_train["time"].values, event=data_train["event"].values)
    # time_bins = torch.cat([time_bins, torch.tensor([data_train["time"].values.max()])], dim=0) # delete
    num_time_bins = len(time_bins)

    if args.model == "MTLR":
        model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=args)
    elif args.model == "BayesianLinearMTLR":
        model = BayesLinMtlr(in_features=num_features, num_time_bins=num_time_bins, config=args)
    elif args.model == "BayesianMTLR":
        model = BayesMtlr(in_features=num_features, num_time_bins=num_time_bins, config=args)
    elif args.model == "BayesianHorseshoeLinearMTLR":
        model = BayesHsLinMtlr(in_features=num_features, num_time_bins=num_time_bins, config=args)
    elif args.model == "BayesianElementwiseMTLR":
        model = BayesEleMtlr(in_features=num_features, num_time_bins=num_time_bins, config=args)
    elif args.model == "BayesianHorseshoeMTLR":
        model = BayesHsMtlr(in_features=num_features, num_time_bins=num_time_bins, config=args)
    elif args.model == "CoxPH":
        model = CoxPH(in_features=num_features, config=args)
    elif args.model == "BayesianLinearCox":
        model = BayesLinCox(in_features=num_features, config=args)
    elif args.model == "BayesianCox":
        model = BayesCox(in_features=num_features, config=args)
    elif args.model == "BayesianHorseshoeLinearCox":
        model = BayesHsLinCox(in_features=num_features, config=args)
    elif args.model == "BayesianElementwiseCox":
        model = BayesEleCox(in_features=num_features, config=args)
    elif args.model == "BayesianHorseshoeCox":
        model = BayesHsCox(in_features=num_features, config=args)
    else:
        raise NotImplementedError

    print(model)
    model = train_model(model, data_train, time_bins, config=args, path=path + f"/exp.pth",
                        random_state=seed, reset_model=True, device=device)

    x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
    if isinstance(model, mtlr):
        survival_outputs, time_bins, ensemble_outputs = make_mtlr_prediction(model, x_test, time_bins, config=args)
    elif isinstance(model, BayesEleMtlr) or isinstance(model, BayesLinMtlr):
        survival_outputs, time_bins, ensemble_outputs = make_ensemble_mtlr_prediction(model, x_test, time_bins,
                                                                                      config=args)
    elif isinstance(model, CoxPH):
        survival_outputs, time_bins, ensemble_outputs = make_cox_prediction(model, x_test, config=args)
    elif isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        survival_outputs, time_bins, ensemble_outputs = make_ensemble_cox_prediction(model, x_test, config=args)
    else:
        raise NotImplementedError

    print('*' * 10 + 'Start Evaluation' + '*' * 10)
    # plot_curve_with_bar(time_bins, survival_outputs, upper_outputs, lower_outputs, index=0)
    eval = BaseEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                         data_train.time.values, data_train.event.values)
    print(eval.d_calibration()[1])

    print("Concordance:", eval.concordance(ties='All')[0])
    print("MAE:", eval.l1_loss(method="Margin"))
    print("D-cal:", eval.d_calibration()[0])
    ## Drop X% of the data method
    coverage_stats = {}
    for percentage in credible_region_sizes:
        drop_num = math.floor(0.5 * args.n_samples_test * (1 - percentage))
        lower_outputs = torch.kthvalue(ensemble_outputs, k=1 + drop_num, dim=0)[0]
        upper_outputs = torch.kthvalue(ensemble_outputs, k=args.n_samples_test - drop_num, dim=0)[0]
        coverage_stats[percentage] = coverage(time_bins, upper_outputs, lower_outputs,
                                              data_test.time.values, data_test.event.values)
    print("C-cal:", coverage_stats)


if __name__ == '__main__':
    main()
