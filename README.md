# Predicting and Auditing Online Purchase Conversion

This project uses the UCI Online Shoppers Purchasing Intention dataset to classify completed online shopping sessions as purchase or non-purchase sessions using supervised machine learning.

The workflow includes data inspection, preprocessing, feature preparation, model training, model comparison, classification evaluation and interpretation of the main conversion signals. The project is framed as a post-session audit, not a live intervention model.

## Data source

Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository.

Citation: C. Sakar and Y. Kastro, "Online Shoppers Purchasing Intention Dataset," UCI Machine Learning Repository, 2018. DOI: 10.24432/C5F88Q.

The dataset is licensed under the Creative Commons Attribution 4.0 International licence.

## Setup and requirements

This notebook requires the package versions listed in `requirements.txt`. In particular, it requires `scikit-learn>=1.8` because the Logistic Regression tuning uses the newer `l1_ratio` regularisation API.

To set up the environment, run:

```bash
pip install -r requirements.txt
```

The notebook includes a version check at the start and will stop with a clear error message if the required scikit-learn version is not installed.

The notebook expects the dataset file to be named:

```text
online_shoppers_intention.csv
```

Place the file in one of these locations:

```text
online_shoppers_intention.csv
data/online_shoppers_intention.csv
../data/online_shoppers_intention.csv
```

The notebook checks these paths automatically and loads the first matching file. If the dataset is not found, it stops with a clear `FileNotFoundError` showing which locations were checked.

For a full reproduction, run the notebook from top to bottom. The notebook saves fitted models and cached result tables in the `outputs/` folder so the later evaluation sections can be rerun without repeating every tuning cell.
