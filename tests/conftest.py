import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta


@pytest.fixture
def sample_df():
    """75-row Polars DataFrame with realistic housing data."""
    rng = np.random.default_rng(42)
    n = 75
    base_date = date(2020, 1, 1)
    dates = [base_date + timedelta(days=int(d)) for d in rng.integers(0, 730, size=n)]
    regions = rng.choice(["east", "west"], size=n).tolist()

    x = rng.uniform(100_000, 900_000, n)
    y = rng.uniform(6_000_000, 7_000_000, n)
    floor_area = rng.uniform(20, 200, n)
    seashore_distance = rng.uniform(0, 50_000, n)

    # Build a target with a real signal so models can learn
    price = (
        1000 * floor_area
        - 0.5 * seashore_distance
        + 0.1 * x
        + rng.normal(0, 10_000, n)
    )
    price = np.clip(price, 50_000, 500_000)

    return pl.DataFrame({
        "x": x,
        "y": y,
        "floor_area": floor_area,
        "transaction_date": dates,
        "seashore_distance": seashore_distance,
        "transaction_amount": price,
        "region": regions,
    })


@pytest.fixture
def target(sample_df):
    return sample_df["transaction_amount"].to_numpy()


@pytest.fixture
def features():
    return ["x", "y", "floor_area", "transaction_date", "seashore_distance"]
