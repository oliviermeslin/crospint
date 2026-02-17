# crospint: Coordinate Rotation for Spatial Interpolation

A scikit-learn-compatible pipeline for spatial and spatio-temporal interpolation using coordinate rotation and ensemble methods.

## Overview

Tree-based models (LightGBM, Random Forest) split data along axis-aligned boundaries, which limits their ability to capture spatial patterns that don't align with the coordinate axes. `crospint` solves this by augmenting the feature set with rotated copies of the geographic coordinates, allowing the model to perform oblique spatial splits.

The package offers full compatibility with `scikit-learn`. It accepts [Polars](https://pola.rs/) DataFrames as input, while maintaining full compatibility with ML libraries that require [Pandas](https://pandas.pydata.org/) DataFrames as input. It also provides `TwoStepsModel`, a housing price estimator that handles log-transformation, price-per-square-meter conversion, retransformation bias correction, and iterative calibration.

## Installation

Install with `pip`:

```bash
pip install crospint
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add crospint
```

**Dependencies:** `polars`, `pandas`, `pyarrow`, `scipy`, `scikit-learn`, `lightgbm`, `matplotlib`.

## Quick start

```python
import numpy as np
import polars as pl
from datetime import date, timedelta
from lightgbm import LGBMRegressor
from crospint import create_model_pipeline

# Generate synthetic data
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

# Create and configure the pipeline
pipe = create_model_pipeline(model=LGBMRegressor(n_estimators=100, verbose=-1))
pipe.set_params(
    coord_rotation__coordinates_names=("x", "y"),
    coord_rotation__number_axis=11,
    date_conversion__date_name="transaction_date",
)

# Train and predict
features = ["x", "y", "floor_area", "transaction_date", "seashore_distance"]
target = df["transaction_amount"].to_numpy()
pipe.fit(df[features], target)
predictions = pipe.predict(df[features])
```

## Documentation

- **[User guide](doc/guide.md)** -- detailed walkthrough of all features: pipeline configuration, housing price modeling, calibration, outlier detection, and API reference.


# Notebooks

Two notebooks illustrate the valuation method describe in Meslin [2026]:

- How to apply the conditional tail removal procedure [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliviermeslin/crospint/blob/improve_readme/notebooks/outlier_removal_through_CTR.ipynb)

- How to train price models at the national scale [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliviermeslin/crospint/blob/improve_readme/notebooks/model_training_and_evaluation.ipynb)


## License

MIT -- see [LICENSE](LICENSE) for details.
