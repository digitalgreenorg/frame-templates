# Solution for [Digital Green Crop Yield Estimate Challenge](https://zindi.africa/competitions/digital-green-crop-yield-estimate-challenge)

My solution is ensemble of 3 CatBoostRegressor models, each of them trained on 5-fold split. Solution focuses on robust cross validation and dealing with outliers, rather than on complex feature engineering. I decided to model yield per acre, rather than yield because of very high correlation between them (~0.9). As post processing, I multiply prediction by acre variable to get original target variable. I also normalize most continuous variables by area. I used most of available variables in the modeling with the exception of "CropTillageDate", "Harv_date", "Threshing_date" which did not seem to contain useful information. When I was choosing hyperparameters, I heavily regularized the model to prevent overfitting.

### Folder layout
    .
    ├── ...
    ├── data
    │   ├── Train.csv
    │   └── Test.csv
    ├── environment.yml
    ├── README.md
    └── run.py


## Setup and usage

Use conda to setup environment

```bash
conda env create -f environment.yml 
conda activate crop_yield
```

Execute run.py script to train models and make prediction.

```bash
python3 run.py
```

Running these commands is going to create solution.csv file that corresponds RMSE score of 429.656 and 102.432 on public and private leaderboard respectively.

## Hardware requirements
No special requirements. Running script on my average laptop takes about 10 minutes.
