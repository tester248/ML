# ML — Practical Assignments

Machine learning practicals implemented in Jupyter Notebook.

| Prac | Topic | Notebook |
| --- | --- | --- |
| 1 | Feature Transformation using LDA | [`prac1/prac1LDA.ipynb`](prac1/prac1LDA.ipynb) |
| 1A | PCA for Dimensionality Reduction | [`prac1a/prac1aPCA.ipynb`](prac1a/prac1aPCA.ipynb) |
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

## Prac 1A — PCA on the wine dataset

Compresses 13 chemical measurements of wine into a small number of principal
components that retain most of the variation, so the customer segments become
visually separable.

- **Source:** [GeeksforGeeks — Wine.csv](https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv) (178 samples, 13 features, 3 segments)
- **Scaling:** applied before fitting PCA. `Proline` (~1000) and `Magnesium` (~100) dwarf `pH` (~3), so unscaled PCA would make PC1 almost purely a magnitude axis rather than a chemical one.
- **Variance:** PC1 36.2%, PC2 19.2% → **2 components hold 55.4%**. Reaching 95% needs 10 of 13.
- **Separability:** segments separate cleanly along PC1, which is dominated by phenolic and colour compounds — Flavanoids (0.423), Total_Phenols (0.395), OD280 (0.376), Proanthocyanins (0.313).
- **Accuracy check:** logistic regression scores **0.9167** on 2 components against **0.9722** on all 13, so discarding 11 dimensions costs about 5.5 points of accuracy.

> The brief describes separating *red* from *white* wine, but the linked dataset
> labels three grape cultivars as `Customer_Segment` 1-3 rather than wine colour.
> The PCA method is unaffected — the label is only used to colour the plots.

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

## Prac 3 — SVM handwritten-digit classification

Classifies handwritten digits from 0 to 9 using linear and RBF-kernel Support
Vector Machines. The notebook includes training-only feature scaling, a
stratified 70/30 train/test split, RBF hyperparameter tuning with three-fold
cross-validation, accuracy and classification metrics, a confusion matrix, and
prediction visualizations.

- **Source:** scikit-learn's built-in `load_digits` dataset (1,797 samples,
  8×8 images, 64 pixel features, 10 classes)
- **Why this source:** it keeps the practical self-contained and avoids
  downloading a large external CSV. The assignment's Kaggle MNIST variant uses
  28×28 images and 784 features; the SVM workflow is otherwise the same.
- **Notebook:** [`prac3/prac3.ipynb`](prac3/prac3.ipynb)

Run it with the repository virtual environment:

```bash
.venv\Scripts\activate
jupyter notebook prac3/prac3.ipynb
```

## Prac 4 — K-Means clustering

Applies K-Means clustering to the Iris dataset. The notebook standardizes the
four measurements, uses the elbow method to select three clusters, visualizes
the clusters with PCA, and reports silhouette score and adjusted Rand index.

- **Source:** [`prac1/Iris.csv`](prac1/Iris.csv)
- **Notebook:** [`prac4/prac4.ipynb`](prac4/prac4.ipynb)

## Prac 5 — Random Forest car-safety classification

Predicts car-evaluation categories using a Random Forest classifier. The
categorical inputs are one-hot encoded, then evaluated with accuracy,
classification metrics, a confusion matrix, and feature importances.

- **Source:** [`prac5/car.data`](prac5/car.data), from the UCI Car Evaluation
  dataset linked in the assignment
- **Notebook:** [`prac5/prac5.ipynb`](prac5/prac5.ipynb)

## Prac 6 — Tic-Tac-Toe reinforcement learning

Builds a Tic-Tac-Toe environment and trains a Q-learning agent against a random
opponent. The notebook covers environment setup, game logic, Q-table updates,
training progress, testing, and a sample game.

- **Notebook:** [`prac6/prac6.ipynb`](prac6/prac6.ipynb)

## Exporting to PDF

Neither notebook renders directly to PDF because LaTeX is not installed. The
committed PDFs were produced by converting to HTML and printing from a browser:

```bash
jupyter nbconvert --to html prac1/prac1LDA.ipynb
```

Then open the HTML, press `Ctrl+P`, choose **Save as PDF**, paper size **A4**,
and enable **Background graphics**. The committed PDFs were produced this way.

Keep source lines under ~85 characters — nbconvert does not wrap long lines, so
anything wider than the page is clipped rather than wrapped.
