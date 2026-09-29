# ML — Practical Assignments

Machine learning practicals implemented in Jupyter Notebook.

| Prac | Topic | Notebook |
| --- | --- | --- |
| 1 | Feature Transformation using LDA | [`prac1/prac1LDA.ipynb`](prac1/prac1LDA.ipynb) |
| 2 | Regression Analysis (Linear, Ridge, Lasso) | [`prac2/prac2.ipynb`](prac2/prac2.ipynb) |

## Setup

Create the virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Launch JupyterLab:

```bash
jupyter lab
```

## Prac 1 — LDA on the Iris dataset

Reduces the four flower measurements to **two linear discriminant components**
(classes − 1 = 3 − 1 = 2) using `LinearDiscriminantAnalysis`, then classifies each
flower by species.

- **Source:** Kaggle — [uciml/iris](https://www.kaggle.com/datasets/uciml/iris) (150 rows, 4 features, 3 species)
- **Result:** 100% test accuracy (30/30), confusion matrix fully diagonal

Setup follows the principle that the train/test split is done *before* scaling, so
the test set never leaks into the scaler's statistics.

## Prac 2 — Uber fare regression

Predicts the fare of an Uber trip from pickup/dropoff coordinates and pickup
datetime. Covers preprocessing, outlier removal, correlation analysis, and three
regression models.

- **Source:** Kaggle — [yasserh/uber-fares-dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) (200,000 rows, 9 columns)
- **Preprocessing:** drop unused columns, handle nulls, expand `pickup_datetime`
  into `hour` / `day` / `month` / `year`, drop non-positive fares → 199,977 rows
- **Outliers:** IQR method on `fare_amount` (bounds −3.75 to 22.25), removing 17,155 rows → 182,822 rows
- **Metrics reported:** R2, Adjusted R2, MAE, MSE, RMSE, RMSLE

| Model | R2 | RMSE |
| --- | --- | --- |
| Linear Regression | 0.0202 | 4.1330 |
| Ridge Regression | 0.0202 | 4.1330 |
| Lasso Regression | 0.0186 | 4.1363 |

> **Note on the low R2.** All three scores sit near zero, and that is a genuine
> property of this feature set rather than a bug. Trip *distance* is never
> computed, and a linear model cannot derive distance from two raw coordinate
> pairs because the relationship is geometric, not linear. The model therefore
> mostly learns that fares drifted upward over time (`year` correlates 0.135 with
> `fare_amount`, every other feature is under 0.03). It also explains why Ridge
> and Lasso land so close to plain Linear Regression: with no multicollinearity
> present, the penalty term has almost nothing to correct.

## Exporting to PDF

Neither notebook renders directly to PDF because LaTeX is not installed. The
committed PDFs were produced by converting to HTML and printing from a browser:

```bash
jupyter nbconvert --to html prac1/prac1LDA.ipynb
```

Then open the HTML, press `Ctrl+P`, choose **Save as PDF**, paper size **A4**,
and enable **Background graphics**.

Keep source lines under ~85 characters — nbconvert does not wrap long lines, so
anything wider than the page is clipped rather than wrapped.
