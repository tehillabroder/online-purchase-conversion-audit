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
python3 -m pip install -r requirements.txt
```

The notebook includes a version check at the start and will stop with a clear error message if the required scikit-learn version is not installed.

If running the notebook through the Codio/Jupyter notebook interface, register the same Python environment as a Jupyter kernel:

```bash
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name aml-online-shoppers --display-name "Python (AML online shoppers)"
```

Then open the notebook and select:

```text
Kernel → Change Kernel → Python (AML online shoppers)
```

This ensures the notebook uses the same Python environment where `requirements.txt` was installed.