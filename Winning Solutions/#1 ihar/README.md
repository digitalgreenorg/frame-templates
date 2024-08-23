## How to use
* Unzip and go to project folder:

`cd digi_green_zindi`

* Create new conda virtual environment:

`conda env create -f environment.yml`

* Activate it:

`conda activate dg`

* Start jupyter notebook:

`jupyter notebook`

* Open file `digital_green.ipynb` and run all cells there.

The results will be saved at the file `submission_class01_5.csv` in the same folder.

## Data sources
Only datasets provided by Zindi were used. They are located in the same folder `digi_green_zindi`.

`Train.csv` - train dataset contains the target column "Yield"

`Test.csv` - test dataset without the target column "Yield"

## How it works
The "Yield" column was normalized to the "Acre" column and used as the target. 
Some minimal feature engineering was implemented, as indicated in the corresponding cell of the script. 
The graph in the script illustrates the relationship between the normalized target column "Yield" 
and the column "Acre," revealing two distinct groups of instances: those with target values 
below 5000 (denoted as class 0) and those with target values above 5000 (denoted as class 1). 
It is likely that the second group is composed of outliers. To classify the test dataset into 
class 0 or 1, the CatBoostClassifier with grid search for hyperparameter tuning was employed. 
Subsequently, CatBoostRegressor models were created for each class, and their predictions 
were combined in the final step.

Optimal parameters of the CatBoostRegressor were found using optuna (https://optuna.org/). 
To reproduce the submitted results you can just run the script with set `use_optuna = False`.
In this case no hyperparameter search will be performed and the script will use the set of parameters
from the former hyperparameter optimization. The running time of the script on 
HP ZBook Fury 15 G8 (Intel(R) Core(TM) i7-11850H CPU @ 2.50GHz) in this case is **2 minutes**.
If you set `use_optuna = True` and `n_jobs = -1` a new parallelized hyperparameter search will be performed.
The number of parallel jobs will be set to CPU count. The running time of the script on 
HP ZBook Fury 15 G8 (Intel(R) Core(TM) i7-11850H CPU @ 2.50GHz) in this case is **1 hour**.
It's important to note that when optimizing hyperparameters in parallel mode, 
there is inherent non-determinism, making it difficult to reproduce the same results. 
Nevertheless, the RMSE of the obtained solution will likely be better than the RMSE of the submitted solution.
If you want to achieve reproducible results of the hyperparameter optimization with optuna, 
you must set the parameter `n_jobs = 1` to perform a sequential search. The running time of the script on 
HP ZBook Fury 15 G8 (Intel(R) Core(TM) i7-11850H CPU @ 2.50GHz) in this case is **6 hours**.
