#!/bin/bash

# Put your absolute path for the SurvivalEVAL folder here
export PYTHONPATH="${PYTHONPATH}:/home/shiang/Documents/GithubRepository/BNN-ISD/SurvivalEVAL"

python3 run_models.py --dataset Synthetic-I --model MTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-I --model BayesianElementwiseMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-I --model BayesianHorseshoeLinearMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-I --model BayesianLinearMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-I --model CoxPH --lr 0.005
python3 run_models.py --dataset Synthetic-I --model BayesianHorseshoeLinearCox --lr 0.005
python3 run_models.py --dataset Synthetic-I --model BayesianLinearCox --lr 0.005
python3 run_models.py --dataset Synthetic-I --model BayesianElementwiseCox --lr 0.005

python3 run_models.py --dataset Synthetic-II --model MTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-II --model BayesianHorseshoeLinearMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-II --model BayesianElementwiseMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-II --model BayesianHorseshoeMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-II --model BayesianLinearMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-II --model BayesianMTLR --lr 0.00008
python3 run_models.py --dataset Synthetic-II --model CoxPH --lr 0.005
python3 run_models.py --dataset Synthetic-II --model BayesianHorseshoeLinearCox --lr 0.005
python3 run_models.py --dataset Synthetic-II --model BayesianElementwiseCox --lr 0.005
python3 run_models.py --dataset Synthetic-II --model BayesianHorseshoeCox --lr 0.005
python3 run_models.py --dataset Synthetic-II --model BayesianLinearCox --lr 0.005
python3 run_models.py --dataset Synthetic-II --model BayesianCox --lr 0.005

python3 run_models.py --dataset SUPPORT --model MTLR --lr 0.00008
python3 run_models.py --dataset SUPPORT --model BayesianElementwiseMTLR --lr 0.00008
python3 run_models.py --dataset SUPPORT --model BayesianHorseshoeMTLR --lr 0.00008
python3 run_models.py --dataset SUPPORT --model BayesianMTLR --lr 0.00008
python3 run_models.py --dataset SUPPORT --model BayesianHorseshoeLinearMTLR --lr 0.00008
python3 run_models.py --dataset SUPPORT --model BayesianLinearMTLR --lr 0.00008
python3 run_models.py --dataset SUPPORT --model CoxPH --lr 0.005
python3 run_models.py --dataset SUPPORT --model BayesianElementwiseCox --lr 0.005
python3 run_models.py --dataset SUPPORT --model BayesianHorseshoeCox --lr 0.005
python3 run_models.py --dataset SUPPORT --model BayesianCox --lr 0.005
python3 run_models.py --dataset SUPPORT --model BayesianHorseshoeLinearCox --lr 0.005
python3 run_models.py --dataset SUPPORT --model BayesianLinearCox --lr 0.005
# Need to apply NACD dataset first
#python3 run_models.py --dataset NACD --model MTLR --lr 0.00008
#python3 run_models.py --dataset NACD --model BayesianElementwiseMTLR --lr 0.00008
#python3 run_models.py --dataset NACD --model BayesianHorseshoeMTLR --lr 0.00008
#python3 run_models.py --dataset NACD --model BayesianMTLR --lr 0.00008
#python3 run_models.py --dataset NACD --model BayesianHorseshoeLinearMTLR --lr 0.00008
#python3 run_models.py --dataset NACD --model BayesianLinearMTLR --lr 0.00008
#python3 run_models.py --dataset NACD --model CoxPH --lr 0.005
#python3 run_models.py --dataset NACD --model BayesianHorseshoeLinearCox --lr 0.005
#python3 run_models.py --dataset NACD --model BayesianLinearCox --lr 0.005
#python3 run_models.py --dataset NACD --model BayesianElementwiseCox --lr 0.005
#python3 run_models.py --dataset NACD --model BayesianHorseshoeCox --lr 0.005
#python3 run_models.py --dataset NACD --model BayesianCox --lr 0.005
# Need to apply MIMIC dataset first
#python3 run_models.py --dataset MIMIC --model MTLR --lr 0.00008
#python3 run_models.py --dataset MIMIC --model BayesianElementwiseMTLR --lr 0.00008
#python3 run_models.py --dataset MIMIC --model BayesianHorseshoeMTLR --lr 0.00008
#python3 run_models.py --dataset MIMIC --model BayesianMTLR --lr 0.00008
#python3 run_models.py --dataset MIMIC --model BayesianHorseshoeLinearMTLR --lr 0.00008
#python3 run_models.py --dataset MIMIC --model BayesianLinearMTLR --lr 0.00008
#python3 run_models.py --dataset MIMIC --model CoxPH --lr 0.005
#python3 run_models.py --dataset MIMIC --model BayesianHorseshoeLinearCox --lr 0.005
#python3 run_models.py --dataset MIMIC --model BayesianLinearCox --lr 0.005
#python3 run_models.py --dataset MIMIC --model BayesianElementwiseCox --lr 0.005
#python3 run_models.py --dataset MIMIC --model BayesianHorseshoeCox --lr 0.005
#python3 run_models.py --dataset MIMIC --model BayesianCox --lr 0.005
