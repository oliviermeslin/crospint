import math
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from datetime import date

from crospint.interpolation import (
    rotate_point,
    ValidateFeatures,
    AddCoordinatesRotation,
    is_valid_ymd,
    ConvertDateToInteger,
    ConvertToPandas,
    create_model_pipeline,
)


# ── rotate_point ──────────────────────────────────────────────────────────


class TestRotatePoint:
    def test_rotate_point_zero_angle(self):
        xx, yy = rotate_point(3.0, 4.0, 0)
        assert xx == pytest.approx(3.0)
        assert yy == pytest.approx(4.0)

    def test_rotate_point_90_degrees(self):
        xx, yy = rotate_point(1.0, 0.0, 90)
        assert xx == pytest.approx(0.0, abs=1e-10)
        assert yy == pytest.approx(1.0)

    def test_rotate_point_180_degrees(self):
        xx, yy = rotate_point(1.0, 0.0, 180)
        assert xx == pytest.approx(-1.0)
        assert yy == pytest.approx(0.0, abs=1e-10)

    def test_rotate_point_360_degrees(self):
        xx, yy = rotate_point(3.0, 4.0, 360)
        assert xx == pytest.approx(3.0)
        assert yy == pytest.approx(4.0)

    def test_rotate_point_custom_center(self):
        # Rotating (6,5) 90° around (5,5) should give (5,6)
        xx, yy = rotate_point(6.0, 5.0, 90, center=(5, 5))
        assert xx == pytest.approx(5.0, abs=1e-10)
        assert yy == pytest.approx(6.0)

    def test_rotate_point_array_input(self):
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        xx, yy = rotate_point(x, y, 90)
        np.testing.assert_allclose(xx, [0.0, -1.0], atol=1e-10)
        np.testing.assert_allclose(yy, [1.0, 0.0], atol=1e-10)


# ── ValidateFeatures ─────────────────────────────────────────────────────


class TestValidateFeatures:
    def test_validate_features_fit_stores_columns(self, sample_df, features):
        vf = ValidateFeatures()
        vf.fit(sample_df.select(features))
        assert vf.feature_names == features
        assert vf.is_fitted is True

    def test_validate_features_transform_reorders(self, sample_df, features):
        vf = ValidateFeatures()
        vf.fit(sample_df.select(features))
        shuffled = sample_df.select(features[::-1])
        result = vf.transform(shuffled)
        assert result.columns == features

    def test_validate_features_missing_column_raises(self, sample_df, features):
        vf = ValidateFeatures()
        vf.fit(sample_df.select(features))
        incomplete = sample_df.select(features[:-1])
        with pytest.raises(ValueError, match="missing"):
            vf.transform(incomplete)

    def test_validate_features_not_polars_raises(self, sample_df, features):
        vf = ValidateFeatures()
        with pytest.raises(AssertionError):
            vf.fit(sample_df.select(features).to_pandas())

    def test_validate_features_extra_columns_ignored(self, sample_df, features):
        vf = ValidateFeatures()
        vf.fit(sample_df.select(features))
        extra = sample_df.select(features + ["region"])
        result = vf.transform(extra)
        assert result.columns == features


# ── AddCoordinatesRotation ────────────────────────────────────────────────


class TestAddCoordinatesRotation:
    def test_rotation_adds_columns(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=4)
        df = sample_df.select(features)
        acr.fit(df)
        result = acr.transform(df)
        # 3 extra rotations × 2 coords = 6 new columns
        assert len(result.columns) == len(features) + 6

    def test_rotation_number_axis_1_no_new_columns(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=1)
        df = sample_df.select(features)
        acr.fit(df)
        result = acr.transform(df)
        assert result.columns == features

    def test_rotation_missing_coords_raises(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=None, number_axis=4)
        with pytest.raises(ValueError, match="coordinates_names"):
            acr.fit(sample_df.select(features))

    def test_rotation_wrong_length_coords_raises(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x",), number_axis=4)
        with pytest.raises(ValueError, match="two coordinates"):
            acr.fit(sample_df.select(features))

    def test_rotation_coord_not_in_df_raises(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x", "z"), number_axis=4)
        with pytest.raises(ValueError, match="not in the data"):
            acr.fit(sample_df.select(features))

    def test_rotation_missing_number_axis_raises(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=None)
        with pytest.raises(ValueError, match="number_axis"):
            acr.fit(sample_df.select(features))

    def test_rotation_center_is_mean(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=4)
        df = sample_df.select(features)
        acr.fit(df)
        assert acr.center[0] == pytest.approx(df["x"].mean())
        assert acr.center[1] == pytest.approx(df["y"].mean())

    def test_rotation_not_polars_raises(self, sample_df, features):
        acr = AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=4)
        with pytest.raises(AssertionError):
            acr.fit(sample_df.select(features).to_pandas())


# ── is_valid_ymd ──────────────────────────────────────────────────────────


class TestIsValidYmd:
    def test_valid_date(self):
        assert is_valid_ymd("2020-01-15") is True

    def test_invalid_format(self):
        assert is_valid_ymd("01/15/2020") is False

    def test_invalid_date(self):
        assert is_valid_ymd("2020-13-01") is False

    def test_valid_list(self):
        assert is_valid_ymd(["2020-01-01", "2021-06-15"]) is True

    def test_invalid_in_list(self):
        assert is_valid_ymd(["2020-01-01", "bad-date"]) is False


# ── ConvertDateToInteger ──────────────────────────────────────────────────


class TestConvertDateToInteger:
    def test_date_to_int_correct_value(self):
        df = pl.DataFrame({
            "d": [date(2010, 1, 2)],
        })
        cdi = ConvertDateToInteger(date_name="d", reference_date="2010-01-01")
        cdi.fit(df)
        result = cdi.transform(df)
        assert result["d"][0] == 1

    def test_date_to_int_reference_date_is_zero(self):
        df = pl.DataFrame({"d": [date(2010, 1, 1)]})
        cdi = ConvertDateToInteger(date_name="d", reference_date="2010-01-01")
        cdi.fit(df)
        result = cdi.transform(df)
        assert result["d"][0] == 0

    def test_date_to_int_invalid_reference_raises(self):
        with pytest.raises(ValueError, match="reference date"):
            ConvertDateToInteger(date_name="d", reference_date="not-a-date")

    def test_date_to_int_missing_column_raises(self):
        df = pl.DataFrame({"other": [date(2020, 1, 1)]})
        cdi = ConvertDateToInteger(date_name="d")
        with pytest.raises(ValueError, match="not in the data"):
            cdi.fit(df)

    def test_date_to_int_wrong_dtype_raises(self):
        df = pl.DataFrame({"d": ["2020-01-01"]})
        cdi = ConvertDateToInteger(date_name="d")
        with pytest.raises(TypeError, match="not of type date"):
            cdi.fit(df)

    def test_date_to_int_not_polars_raises(self):
        import pandas as pd
        pdf = pd.DataFrame({"d": [pd.Timestamp("2020-01-01")]})
        cdi = ConvertDateToInteger(date_name="d")
        with pytest.raises(AssertionError):
            cdi.fit(pdf)


# ── ConvertToPandas ───────────────────────────────────────────────────────


class TestConvertToPandas:
    def test_to_pandas_returns_pandas(self, sample_df, features):
        ctp = ConvertToPandas()
        ctp.fit(sample_df.select(features))
        result = ctp.transform(sample_df.select(features))
        assert isinstance(result, pd.DataFrame)

    def test_to_pandas_categoricals(self):
        df = pl.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
        ctp = ConvertToPandas()
        ctp.fit(df)
        result = ctp.transform(df)
        assert isinstance(result["cat"].dtype, pd.CategoricalDtype)

    def test_to_pandas_not_fitted_raises(self):
        df = pl.DataFrame({"a": [1]})
        ctp = ConvertToPandas()
        with pytest.raises(RuntimeError, match="not been fitted"):
            ctp.transform(df)

    def test_to_pandas_not_polars_raises(self):
        ctp = ConvertToPandas()
        with pytest.raises(TypeError, match="Polars"):
            ctp.fit(pd.DataFrame({"a": [1]}))

    def test_to_pandas_handles_nulls(self):
        df = pl.DataFrame({"cat": ["a", None, "b"]})
        ctp = ConvertToPandas()
        ctp.fit(df)
        result = ctp.transform(df.with_columns(pl.col("cat").fill_null("missing")))
        assert "missing" in result["cat"].cat.categories.tolist()


# ── create_model_pipeline ─────────────────────────────────────────────────


class TestCreateModelPipeline:
    def test_pipeline_default_has_5_steps(self):
        pipe = create_model_pipeline()
        assert len(pipe.steps) == 5

    def test_pipeline_no_coords_has_4_steps(self):
        pipe = create_model_pipeline(presence_coordinates=False)
        assert len(pipe.steps) == 4

    def test_pipeline_no_date_has_4_steps(self):
        pipe = create_model_pipeline(presence_date=False)
        assert len(pipe.steps) == 4

    def test_pipeline_neither_has_3_steps(self):
        pipe = create_model_pipeline(presence_coordinates=False, presence_date=False)
        assert len(pipe.steps) == 3

    def test_pipeline_custom_model(self):
        lr = LinearRegression()
        pipe = create_model_pipeline(model=lr)
        assert pipe.steps[-1][1] is lr
