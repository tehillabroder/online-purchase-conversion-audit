# Predicting and Auditing Online Purchase Conversion

Applied machine learning analysis of the UCI Online Shoppers Purchasing Intention dataset, using completed session records to distinguish purchase from non-purchase sessions.

The focus is not simply on maximising a single score: the project examines class imbalance, model comparison, feature engineering, precision–recall trade-offs and the limits of interpreting an outcome-adjacent feature.

## What I investigated

The source dataset contains 12,330 completed browsing sessions with `Revenue` as the binary purchase target. The notebook removes 125 exact duplicate non-purchase rows before modelling, leaving 12,205 observations, of which approximately 15.6% are purchases.

This imbalance makes accuracy alone a poor selection criterion. The analysis therefore focuses primarily on purchase-class precision, recall and F1.

The project is deliberately framed as a **post-session conversion audit**, rather than a real-time intervention model. This distinction matters because the strongest feature, `PageValues`, is available in the completed session record but is closely related to the purchase outcome itself.

## Approach

* Preprocessing contained within scikit-learn `Pipeline` and `ColumnTransformer` objects
* Standardisation of numerical features and one-hot encoding of categorical features
* Initial comparison of eight classification algorithms
* Hyperparameter tuning with `RandomizedSearchCV` and stratified five-fold cross-validation
* F1-based tuning to balance precision and recall for the minority purchase class
* Comparison of original and engineered feature sets
* Repeated stratified cross-validation as a stability check
* Operating-threshold sensitivity using out-of-fold training predictions
* Logistic Regression coefficients and permutation importance for interpretation

Hyperparameters are selected from training-set cross-validation rather than from the test results. The held-out split is used as a comparative evaluation across the candidate model designs.

## Results

At the default classification threshold, the two final ensemble models show a useful operating trade-off:

| Model                   | Precision | Recall |    F1 |
| ----------------------- | --------: | -----: | ----: |
| Tuned Gradient Boosting |     0.715 |  0.644 | 0.678 |
| Tuned Random Forest     |     0.668 |  0.696 | 0.682 |

**Gradient Boosting** is retained as the main focused audit model. It produces more precise purchase flags and fewer false positives: 98 on the held-out test set, compared with 132 for Random Forest.

**Random Forest** is retained as the recall-focused alternative. It identifies 266 of the 382 purchase sessions in the test set, compared with 246 for Gradient Boosting, but produces a larger review list and shows a substantially larger train–validation gap in the repeated-CV stability check.

The marginal difference in F1 is therefore less informative than the operating trade-off between the two models.

Feature engineering was also tested rather than assumed to help. The engineered models did not materially improve on the strongest original-feature models, so the final comparison retains the original feature set.

## The `PageValues` boundary

`PageValues` is by far the dominant feature in this dataset, but it is also **outcome-adjacent**. It should not be interpreted as a clean pre-purchase behavioural predictor.

That is acceptable for the completed-session audit used here, where the full finished session record is available. It would not be an appropriate basis for claiming a genuine real-time purchase-intention model.

The dependence is substantial: removing `PageValues` from an untuned Gradient Boosting diagnostic reduced mean cross-validated F1 from **0.657 to 0.138**, while recall fell from **0.598 to 0.079**.

For that reason, the interpretation explicitly separates `PageValues` from the secondary signals around it. These include `Month`, `ExitRates`, `BounceRates` and product-browsing depth.

## Repository contents

* `aml_online_shoppers.ipynb` — full analysis, modelling and interpretation
* `baseline_results_df.csv` — baseline classifier results
* `tuned_results_df.csv` — tuned original-feature results
* `engineered_tuned_results_df.csv` — tuned engineered-feature results
* saved `joblib` model/search artefacts
* `requirements.txt` — Python dependencies

The notebook is intentionally the full analysis artefact; the README summarises the modelling decisions and main results.

## Running the notebook

Install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Then open:

```text
aml_online_shoppers.ipynb
```

in Jupyter.

The notebook looks for `online_shoppers_intention.csv` in the repository root, a `data/` directory, or the parent `data/` directory.

## Data

**Online Shoppers Purchasing Intention Dataset**
C. Sakar and Y. Kastro, UCI Machine Learning Repository, 2018.
DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)

The dataset is distributed under the [Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
