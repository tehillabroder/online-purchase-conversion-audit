# Predicting and Auditing Online Purchase Conversion

This project analyses completed online shopping sessions from the UCI Online Shoppers Purchasing Intention dataset to classify purchase and non-purchase sessions and examine the signals associated with conversion.

The analysis covers imbalanced tabular classification, pipeline-based preprocessing, model comparison, hyperparameter tuning, feature engineering, threshold sensitivity and model interpretation using scikit-learn.

## What I investigated

The source dataset contains 12,330 completed shopping sessions. After removing 125 exact duplicate rows, the modelling dataset contains 12,205 sessions.

`Revenue` is the binary target and approximately 15.6 per cent of sessions result in a purchase. This class imbalance makes accuracy alone a weak measure of model quality, so evaluation gives greater weight to precision, recall and F1 for the purchase class.

The project uses completed session records as a post-session conversion audit. This framing is important because the dataset includes `PageValues`, which is highly informative but closely connected to the eventual purchase outcome.

## Approach

The modelling workflow includes

* preprocessing inside scikit-learn pipelines
* standardisation of numerical features where appropriate
* one-hot encoding of categorical features
* comparison of eight baseline classifiers
* hyperparameter tuning with `RandomizedSearchCV`
* F1-based model selection for the minority purchase class
* class weighting where supported
* feature engineering based on browsing depth and product activity
* repeated stratified cross-validation for stability checks
* operating-threshold sensitivity analysis
* Logistic Regression coefficients for signed interpretation
* permutation importance for the leading ensemble models

Hyperparameters are selected using cross-validation on the training data. The held-out test split is then used to compare fitted model behaviour on unseen sessions.

## Results

The strongest final candidates were Gradient Boosting and Random Forest using the original feature set.

| Model                   | Precision | Recall | F1    |
| ----------------------- | --------- | ------ | ----- |
| Tuned Gradient Boosting | 0.715     | 0.644  | 0.678 |
| Tuned Random Forest     | 0.668     | 0.696  | 0.682 |

Tuned Gradient Boosting is the main audit model. At the default threshold it produces 98 false positives and a review list of 344 sessions. Its repeated cross-validation train to validation F1 gap is 0.071.

Tuned Random Forest is the recall-oriented alternative. It identifies 266 of the 382 purchase sessions in the test set, compared with 246 for Gradient Boosting. This comes with 132 false positives, a review list of 398 sessions and a larger train to validation F1 gap of 0.240.

The small difference in F1 therefore sits alongside a meaningful operating trade-off. Gradient Boosting gives a more selective review list with higher precision, while Random Forest captures more purchase sessions.

A separate threshold analysis for Gradient Boosting selected a threshold of 0.35 using out-of-fold training predictions. On the test set this increased recall from 0.644 to 0.723 and F1 from 0.678 to 0.685, with precision falling from 0.715 to 0.651.

Feature engineering was also tested across the leading model families. The engineered features carried useful signal but did not improve the strongest original-feature models, so the final models retain the original feature set.

## The PageValues boundary

`PageValues` is the dominant feature in this dataset and is closely connected to the purchase outcome.

Its use fits the completed-session audit setting because it is available in the finished session record. Its interpretation belongs to post-session analysis. A pre-purchase behavioural model would require a feature set built only from information available before conversion.

The dependence is substantial. Removing `PageValues` from an untuned Gradient Boosting diagnostic reduced mean cross-validated F1 from 0.657 to 0.138 and recall from 0.598 to 0.079.

The interpretation therefore also examines the secondary signals around `PageValues`. These include

* `Month`
* `ExitRates`
* `BounceRates`
* `ProductRelated`
* `ProductRelated_Duration`

These features give a broader view of timing, exit behaviour and product-browsing depth across completed sessions.

## Repository contents

`aml_online_shoppers.ipynb` contains the full analysis, modelling workflow and interpretation.

The repository also includes

* baseline model results
* tuned original-feature results
* tuned engineered-feature results
* saved joblib model artefacts
* `requirements.txt`

The notebook remains the full analysis artefact, while this README summarises the main modelling decisions and results.

## Running the notebook

Install the required packages

```bash
python3 -m pip install -r requirements.txt
```

The notebook requires `scikit-learn>=1.8` because the Logistic Regression tuning uses the current `l1_ratio` regularisation API.

Then open

```text
aml_online_shoppers.ipynb
```

in Jupyter and run the notebook from the top.

The notebook searches for `online_shoppers_intention.csv` in the repository root, a `data` directory or the parent `data` directory.

## Data

The project uses the UCI Machine Learning Repository [Online Shoppers Purchasing Intention Dataset](//doi.org/10.24432/C5F88Q).

C. O. Sakar and Y. Kastro
Online Shoppers Purchasing Intention Dataset
UCI Machine Learning Repository
2018
DOI `10.24432/C5F88Q`

The dataset is distributed under the [Creative Commons Attribution 4.0 International licence](//creativecommons.org/licenses/by/4.0/).