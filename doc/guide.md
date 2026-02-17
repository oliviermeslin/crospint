# crospint User Guide

## Table of contents

1. [Overview](#1-overview)
2. [Getting started](#2-getting-started)
3. [Housing price modeling with TwoStepsModel](#3-housing-price-modeling-with-twostepsmodel)
4. [Calibration](#4-calibration)
5. [Outlier detection](#5-outlier-detection)
6. [API reference](#6-api-reference)

---

## 1. Overview

### The problem with axis-aligned splits

Tree-based models (LightGBM, Random Forest, XGBoost) partition the feature space using axis-aligned splits. When predicting a spatially distributed variable, this means the model can only draw decision boundaries parallel to the coordinate axes. Real-world spatial patterns rarely align with north-south / east-west directions, so the model needs many splits to approximate diagonal or curved boundaries.

### Coordinate rotation as a solution

`crospint` augments the feature set with **rotated copies** of the geographic coordinates. For example, with `number_axis=8`, the original (x, y) pair is rotated by 45, 90, 135, 180, 225, 270, and 315 degrees around the data centroid, producing 7 additional (x, y) pairs (16 features total). This gives the tree model access to axis-aligned splits in multiple directions, greatly improving spatial interpolation quality.

### Pipeline architecture

`crospint` builds on scikit-learn's `Pipeline`. The standard pipeline created by `create_model_pipeline` has this structure:

```
ValidateFeatures -> AddCoordinatesRotation -> ConvertDateToInteger -> ConvertToPandas -> Model
```

Each step is a scikit-learn transformer (or estimator for the final model). Steps can be configured via `set_params` using the standard `stepname__param` syntax, or disabled by setting `presence_coordinates=False` or `presence_date=False`.

---

## 2. Getting started

### Synthetic dataset

All examples in this guide use the following synthetic dataset. It simulates real-estate transactions with geographic coordinates, floor area, transaction date, distance to the seashore, and a transaction amount.

```python
import numpy as np
import polars as pl
from datetime import date, timedelta

np.random.seed(42)
n = 500
x = np.random.uniform(100_000, 900_000, n)
y = np.random.uniform(6_000_000, 7_100_000, n)
floor_area = np.random.uniform(20, 250, n).astype(int)
seashore_distance = np.random.uniform(0, 500_000, n)
start = date(2018, 1, 1)
days_range = (date(2023, 12, 31) - start).days
transaction_date = [start + timedelta(days=int(d)) for d in np.random.uniform(0, days_range, n)]
price_per_sqm = (2000 + 500 * np.sin(x / 200_000)
                 + 300 * np.cos(y / 300_000)
                 - 0.0005 * seashore_distance
                 + np.random.normal(0, 200, n))
transaction_amount = (floor_area * np.clip(price_per_sqm, 500, None)).round(0)

df = pl.DataFrame({
    "x": x, "y": y, "floor_area": floor_area,
    "transaction_date": transaction_date,
    "seashore_distance": seashore_distance,
    "transaction_amount": transaction_amount,
}).cast({"transaction_date": pl.Date})
```

### Creating a pipeline with `create_model_pipeline`

```python
from lightgbm import LGBMRegressor
from crospint import create_model_pipeline

pipe = create_model_pipeline(model=LGBMRegressor(n_estimators=100, verbose=-1))
```

### Configuring the pipeline with `set_params`

Pipeline steps are configured using scikit-learn's `stepname__param` syntax:

```python
pipe.set_params(
    coord_rotation__coordinates_names=("x", "y"),
    coord_rotation__number_axis=8,
    date_conversion__date_name="transaction_date",
)
```

### Fitting and predicting

```python
features = ["x", "y", "floor_area", "transaction_date", "seashore_distance"]
target = df["transaction_amount"].to_numpy()

pipe.fit(df[features], target)
predictions = pipe.predict(df[features])
```

### Swapping the model

Any scikit-learn-compatible regressor can be used as the final step:

```python
from sklearn.ensemble import RandomForestRegressor

pipe_rf = create_model_pipeline(model=RandomForestRegressor(n_estimators=100, random_state=42))
pipe_rf.set_params(
    coord_rotation__coordinates_names=("x", "y"),
    coord_rotation__number_axis=8,
    date_conversion__date_name="transaction_date",
)
pipe_rf.fit(df[features], target)
```

### Disabling optional steps

If your data has no geographic coordinates or no date column, disable the corresponding steps:

```python
# No coordinates, no date
pipe_simple = create_model_pipeline(
    model=LGBMRegressor(n_estimators=100, verbose=-1),
    presence_coordinates=False,
    presence_date=False,
)
pipe_simple.fit(df[["floor_area", "seashore_distance"]], target)
```

---

## 3. Housing price modeling with TwoStepsModel

`TwoStepsModel` wraps the pipeline and adds:

- **Log-transformation** of the target (with retransformation bias correction).
- **Price-per-square-meter** conversion (divides the target by floor area before fitting, multiplies back after prediction).
- **Validation set** support with early stopping (LightGBM).
- **Calibration** (see [section 4](#4-calibration)).

### Creating and configuring the model

```python
from crospint import TwoStepsModel
from lightgbm import LGBMRegressor

model = TwoStepsModel(
    model=LGBMRegressor(n_estimators=500, verbose=-1),
    log_transform=True,
    price_sq_meter=True,
    floor_area_name="floor_area",
    presence_coordinates=True,
    presence_date=True,
)
```

### Setting pipeline parameters

`TwoStepsModel.set_params` takes a dictionary:

```python
model.set_params({
    "coord_rotation__coordinates_names": ("x", "y"),
    "coord_rotation__number_axis": 8,
    "date_conversion__date_name": "transaction_date",
})
```

### Train / validation / test split

```python
from sklearn.model_selection import train_test_split

features = ["x", "y", "floor_area", "transaction_date", "seashore_distance"]
target = df["transaction_amount"].to_numpy()

df_train, df_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42
)
df_train, df_val, y_train, y_val = train_test_split(
    df_train, y_train, test_size=0.25, random_state=42
)
```

### Fitting with a validation set

When a validation set is provided, LightGBM uses it for early stopping:

```python
model.fit(
    df_train[features], y_train,
    model_features=features,
    X_val=df_val[features], y_val=y_val,
    early_stopping_rounds=25,
    verbose=True,
)
```

### Predicting with retransformation correction

When the target is log-transformed, the naive back-transformation `exp(prediction)` is biased downward. `TwoStepsModel` supports two correction methods:

- **Miller (1984):** multiplies predictions by `exp(RMSE^2 / 2)`.
- **Duan (1983):** multiplies predictions by the mean of `exp(residuals)` (smearing factor).

```python
# Predict with Miller's correction
y_pred = model.predict(
    df_test[features],
    add_retransformation_correction=True,
    retransformation_method="Miller",
)
```

### Key attributes after fitting

| Attribute | Description |
|---|---|
| `model.pipe` | The underlying scikit-learn `Pipeline` |
| `model.preprocessor` | All pipeline steps except the final model |
| `model.model` | The final model (e.g. `LGBMRegressor`) |
| `model.RMSE` | RMSE of the model on transformed scale (when `log_transform=True`) |
| `model.smearing_factor` | Duan's smearing factor (when `log_transform=True`) |
| `model.is_model_fitted` | Whether the model has been fitted |
| `model.model_features` | List of feature names used for fitting |

---

## 4. Calibration

### Why calibrate?

Even with retransformation correction, predicted totals may not match observed totals across important subgroups (e.g. by region, property type). Calibration adjusts predictions so that they are consistent with known marginal totals.

`crospint` provides two calibration steps:

1. **Iterative raking** (`calibrate_model`): computes calibration ratios by iteratively adjusting predictions to match marginal totals of specified variables, optionally combined with distributional calibration via isotonic regression.
2. **Calibration model** (`train_calibration_model`): trains a secondary model to predict the calibration ratios, so they can be applied to new data at prediction time.

### Step 1: Iterative calibration with `calibrate_model`

```python
from crospint.housing_prices import calibrate_model

# Add a categorical variable for calibration
df_val_cal = df_val.with_columns(
    region=pl.when(pl.col("x") < 500_000).then(pl.lit("west")).otherwise(pl.lit("east"))
)

model, converged = calibrate_model(
    model,
    X=df_val_cal[features + ["region"]],
    y=y_val,
    calibration_variables=["region"],
    perform_distributional_calibration=True,
    convergence_rate=1e-3,
    bounds=(0.5, 1.5),
    max_iter=100,
)
```

**Parameters:**

| Parameter | Description | Default |
|---|---|---|
| `model` | A fitted `TwoStepsModel` | -- |
| `X` | Calibration features (Polars DataFrame) | validation set if available |
| `y` | Calibration target (numpy array) | validation target if available |
| `calibration_variables` | List of column names for marginal calibration | `None` |
| `perform_distributional_calibration` | Use isotonic regression for distributional calibration | `True` |
| `convergence_rate` | Stop when max relative gap is below this | `1e-3` |
| `bounds` | Clip calibration ratios to (lower, upper) | `(0.5, 1.5)` |
| `max_iter` | Maximum number of iterations | `100` |

### Step 2: Train a calibration model with `train_calibration_model`

```python
from crospint.housing_prices import train_calibration_model

model = train_calibration_model(
    model,
    calibration_model=LGBMRegressor(
        n_estimators=100, num_leaves=1023, max_depth=12,
        learning_rate=0.5, min_child_samples=20, max_bins=10000,
        random_state=123456, verbose=-1,
    ),
    verbose=True,
)
```

### Predicting with calibration

Once calibrated, use `retransformation_method="calibration"`:

```python
df_test_cal = df_test.with_columns(
    region=pl.when(pl.col("x") < 500_000).then(pl.lit("west")).otherwise(pl.lit("east"))
)

y_pred_calibrated = model.predict(
    df_test_cal[features + ["region"]],
    add_retransformation_correction=True,
    retransformation_method="calibration",
)
```

---

## 5. Outlier detection

A common approach for outlier detection in spatial modeling is to compare observed values against out-of-bag (OOB) predictions from a Random Forest, then flag observations whose residuals are extreme.

> **Note:** This section uses `scipy`, which is **not** a dependency of `crospint`. Install it separately if needed: `pip install scipy`.

### Fit a Random Forest with OOB predictions

```python
from sklearn.ensemble import RandomForestRegressor
from crospint import create_model_pipeline

pipe_oob = create_model_pipeline(
    model=RandomForestRegressor(n_estimators=200, oob_score=True, random_state=42),
)
pipe_oob.set_params(
    coord_rotation__coordinates_names=("x", "y"),
    coord_rotation__number_axis=8,
    date_conversion__date_name="transaction_date",
)
pipe_oob.fit(df[features], target)
```

### Extract OOB predictions and compute residuals

```python
oob_predictions = pipe_oob["model"].oob_prediction_
residuals = np.log(target) - np.log(oob_predictions)
```

### Fit an asymmetric Laplace distribution and compute thresholds

```python
from scipy.stats import laplace_asymmetric

# Fit the distribution to the residuals
params = laplace_asymmetric.fit(residuals)

# Compute thresholds at the 1st and 99th percentiles
lower_threshold = laplace_asymmetric.ppf(0.01, *params)
upper_threshold = laplace_asymmetric.ppf(0.99, *params)
```

### Filter outliers

```python
is_outlier = (residuals < lower_threshold) | (residuals > upper_threshold)
df_clean = df.filter(~pl.Series(is_outlier))
print(f"Removed {is_outlier.sum()} outliers out of {len(df)} observations")
```

---

## 6. API reference

### `create_model_pipeline`

```python
crospint.create_model_pipeline(model=LGBMRegressor(), presence_coordinates=True, presence_date=True)
```

Create a scikit-learn `Pipeline` for spatio-temporal modeling.

| Parameter | Type | Description |
|---|---|---|
| `model` | estimator | Final estimator in the pipeline. Default: `LGBMRegressor()`. |
| `presence_coordinates` | `bool` | Include the `AddCoordinatesRotation` step. Default: `True`. |
| `presence_date` | `bool` | Include the `ConvertDateToInteger` step. Default: `True`. |

**Returns:** `sklearn.pipeline.Pipeline`

```python
pipe = create_model_pipeline(model=LGBMRegressor(verbose=-1))
```

---

### `ValidateFeatures`

```python
crospint.ValidateFeatures()
```

Transformer that validates that all features seen during `fit` are present at `transform` time, and reorders columns to match the training order. Expects Polars DataFrames.

```python
vf = ValidateFeatures()
vf.fit(df[["x", "y"]])
vf.transform(df[["y", "x"]])  # reordered to ["x", "y"]
```

---

### `AddCoordinatesRotation`

```python
crospint.AddCoordinatesRotation(coordinates_names=None, number_axis=None)
```

Transformer that adds rotated copies of geographic coordinates.

| Parameter | Type | Description |
|---|---|---|
| `coordinates_names` | `tuple` | Column names for (x, y) coordinates. |
| `number_axis` | `int` | Total number of axes (original + rotated). E.g. `8` produces 7 rotations. |

**Attributes after fit:** `center` (centroid used for rotation), `rotated_coordinates_names` (list of all coordinate column names).

```python
rot = AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=8)
rot.fit(df)
df_rotated = rot.transform(df)
```

---

### `ConvertDateToInteger`

```python
crospint.ConvertDateToInteger(date_name=None, reference_date="2010-01-01")
```

Transformer that converts a Polars `Date` column to an integer (days since the reference date).

| Parameter | Type | Description |
|---|---|---|
| `date_name` | `str` | Name of the date column. |
| `reference_date` | `str` | Reference date in `YYYY-MM-DD` format. Default: `"2010-01-01"`. |

```python
conv = ConvertDateToInteger(date_name="transaction_date")
conv.fit(df)
df_int = conv.transform(df)  # transaction_date is now an integer
```

---

### `ConvertToPandas`

```python
crospint.ConvertToPandas()
```

Transformer that converts a Polars DataFrame to a pandas DataFrame. String columns are encoded as `pd.Categorical` using categories learned during `fit`, ensuring consistent encoding between training and prediction.

```python
cp = ConvertToPandas()
cp.fit(df)
df_pd = cp.transform(df)  # returns a pandas DataFrame
```

---

### `TwoStepsModel`

```python
crospint.TwoStepsModel(
    model=LGBMRegressor(),
    log_transform=None,
    price_sq_meter=None,
    presence_coordinates=True,
    presence_date=True,
    floor_area_name=None,
)
```

Housing price estimator that wraps `create_model_pipeline` with target transformation (log, price per square meter) and retransformation bias correction.

| Parameter | Type | Description |
|---|---|---|
| `model` | estimator | Model to use in the pipeline. Default: `LGBMRegressor()`. |
| `log_transform` | `bool` | Log-transform the target. |
| `price_sq_meter` | `bool` | Divide target by floor area before fitting. |
| `floor_area_name` | `str` | Column name for floor area (required if `price_sq_meter=True`). |
| `presence_coordinates` | `bool` | Include coordinate rotation step. |
| `presence_date` | `bool` | Include date conversion step. |

**Key methods:**

| Method | Description |
|---|---|
| `set_params(dico)` | Set pipeline parameters via a dictionary. |
| `fit(X, y, model_features=None, X_val=None, y_val=None, ...)` | Fit the model. |
| `predict(X, add_retransformation_correction=False, retransformation_method=None, ...)` | Predict. `retransformation_method` can be `"Miller"`, `"Duan"`, or `"calibration"`. |

```python
m = TwoStepsModel(log_transform=True, price_sq_meter=True, floor_area_name="floor_area")
```

---

### `calibrate_model`

```python
crospint.housing_prices.calibrate_model(
    model, X=None, y=None, calibration_variables=None,
    perform_distributional_calibration=True, convergence_rate=1e-3,
    bounds=(0.5, 1.5), max_iter=100, verbose=True,
)
```

Perform iterative raking calibration on a fitted `TwoStepsModel`. Returns `(model, converged)` where `converged` is a boolean.

---

### `train_calibration_model`

```python
crospint.housing_prices.train_calibration_model(
    model, calibration_model=LGBMRegressor(...), validation_share=0,
    evaluation_period=5, r2_threshold=1, early_stopping_rounds=10, verbose=True,
)
```

Train a secondary model to predict calibration ratios computed by `calibrate_model`. After training, `model.predict(..., retransformation_method="calibration")` applies the learned calibration.

---

### `create_calibration_pipeline`

```python
crospint.create_calibration_pipeline(model=LGBMRegressor())
```

Create a simplified pipeline (no coordinate rotation or date conversion) for the calibration model. Used internally by `train_calibration_model`.

---

### `compute_calibration_ratios`

```python
crospint.compute_calibration_ratios(
    X=None, calibration_variables=None, perform_distributional_calibration=True,
    raw_prediction_variable="predicted_price", cal_prediction_variable="predicted_price_cal",
    target_variable="target", bounds=(0.5, 1.5),
)
```

Compute one round of calibration ratios (distributional and/or marginal). Used internally by `calibrate_model`.
