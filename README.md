# BNN-ISD
Offical Code Release for "Using Bayesian Neural Networks to Select Features and Compute Credible Intervals for Personalized Survival Prediction"

This model can be used for (1) select relevant features for survival analysis, and (2) compute credible intervals for the predicted survival curve.
This model also excels in both discrimination and calibration performance, allowing it to accurately predict the survival curve for individuals.

## Environment Setup
Please install the following packages before running the code:
- Python>=3.7
- CPU or GPU
- Other packages can be installed with the following instruction:
    ```    
    $ pip install -r requirements.txt
    ```
  
## Quick start

### Make Folder/Packages as Source Code
Before running the code, please make sure that the `SurvivalEVAL` folder is treated as source code
- For Pycharm (or other IDEs)):
    - Right-click on the `SurvivalEVAL` folder
    - Select `Mark Directory as` -> `Sources Root`
- For terminal:
    - Add the following lines to your `~/.bashrc` file:
        ```
        export PYTHONPATH="${PYTHONPATH}:/your-path-to/SurvivalEVAL"
        ```
    - Run the following command:
        ```
        $ source ~/.bashrc
        ```
### Running
Running the code with the following command.
```
$ python3 run_models.py --dataset Synthetic-II --model BayesianHorseshoeMTLR --lr 0.00008

```
Or run the bash script:
```
$ bash run.sh
```

## Datasets
The datasets we tested in the paper are `Synthetic-I`, `Synthetic-II`, `SUPPORT`, `NACD`, and `MIMIC`.

For `Synthetic-I` and `Synthetic-II`, we generated them using the code in `datasets.py`.
For `SUPPORT`, we directly download and process the data from the website, which can also be found in `datasets.py`.
That means user can directly run the code on these three datasets.

For `NACD` and `MIMIC`, we cannot directly provide the data due to the privacy issue.

If you are interested in using the `NACD` dataset you can access the NACD data 
from the [Patient Specific Survival Prediction (PSSP) website](http://pssp.srv.ualberta.ca/) under "Public Predictors" or use this [direct download link](http://pssp.srv.ualberta.ca/system/predictors/datasets/000/000/032/original/All_Data_updated_may2011_CLEANED.csv?1350302245). 
And then rename the file to `NACD_Full.csv` and put it in the `data/NACD/` folder.

If you are interested in using the `MIMIC` dataset, you can access the MIMIC data from the [MIMIC website](https://mimic.physionet.org/) under "Accessing MIMIC-IV v2.0", or directly access this [MIMIC-IV Version 2.0](https://physionet.org/content/mimiciv/2.0/).
You first need to go through the ethic process, and once you have done that, you can go to the BigQuery and process the data using the json script `MIMIC_IV_V2.0.json` in the `data/MIMIC/` folder.
And further process the data using the code in `MIMIC_IV_V2.0_preprocess.py`.
