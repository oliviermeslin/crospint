import numpy as np
import polars as pl
import pytest
import lightgbm
from datetime import date

from crospint.housing_prices import (
    TwoStepsModel,
    calibrate_model,
    train_calibration_model,
    create_calibration_pipeline,
    r2_eval,
)


def _make_model(**kwargs):
    """Helper to create a TwoStepsModel with fast LGBMRegressor defaults."""
    defaults = dict(
        model=lightgbm.LGBMRegressor(n_estimators=50, verbose=-1),
        presence_coordinates=True,
        presence_date=True,
    )
    defaults.update(kwargs)
    return TwoStepsModel(**defaults)


def _fit_model(sample_df, target, features, **model_kwargs):
    """Helper to fit a model on the sample data."""
    m = _make_model(**model_kwargs)
    m.set_params({
        "coord_rotation__coordinates_names": ("x", "y"),
        "coord_rotation__number_axis": 2,
        "date_conversion__date_name": "transaction_date",
    })
    m.fit(
        sample_df,
        target,
        model_features=features,
        verbose=False,
        early_stopping_rounds=5,
    )
    return m


# ── TwoStepsModel init ───────────────────────────────────────────────────


class TestTwoStepsModelInit:
    def test_init_default(self):
        m = _make_model()
        assert m.is_model_fitted is False

    def test_init_price_sq_meter_no_floor_area_raises(self):
        with pytest.raises(ValueError, match="floor area"):
            _make_model(price_sq_meter=True, floor_area_name=None)

    def test_init_no_coords_no_date(self):
        m = _make_model(presence_coordinates=False, presence_date=False)
        assert len(m.pipe.steps) == 3


# ── transform / inverse_transform ────────────────────────────────────────


class TestTransformInverse:
    def test_transform_log(self, sample_df, target):
        m = _make_model(log_transform=True)
        result = m.transform(sample_df, target.copy())
        np.testing.assert_allclose(result, np.log(target))

    def test_transform_price_sq_meter(self, sample_df, target):
        m = _make_model(price_sq_meter=True, floor_area_name="floor_area")
        result = m.transform(sample_df, target.copy())
        expected = target / sample_df["floor_area"].to_numpy()
        np.testing.assert_allclose(result, expected)

    def test_transform_both(self, sample_df, target):
        m = _make_model(log_transform=True, price_sq_meter=True, floor_area_name="floor_area")
        result = m.transform(sample_df, target.copy())
        expected = np.log(target / sample_df["floor_area"].to_numpy())
        np.testing.assert_allclose(result, expected)

    def test_inverse_transform_roundtrip(self, sample_df, target):
        m = _make_model(log_transform=True, price_sq_meter=True, floor_area_name="floor_area")
        transformed = m.transform(sample_df, target.copy())
        recovered = m.inverse_transform(sample_df, transformed)
        np.testing.assert_allclose(recovered, target, rtol=1e-10)


# ── fit ───────────────────────────────────────────────────────────────────


class TestFit:
    def test_fit_basic(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features)
        assert m.is_model_fitted is True
        assert m.model_features is not None

    def test_fit_not_polars_raises(self, sample_df, target, features):
        m = _make_model()
        with pytest.raises(AssertionError):
            m.fit(sample_df.to_pandas(), target, model_features=features)

    def test_fit_with_validation(self, sample_df, target, features):
        n = len(sample_df)
        split = int(n * 0.7)
        X_train = sample_df[:split]
        X_val = sample_df[split:]
        y_train = target[:split]
        y_val = target[split:]

        m = _make_model(log_transform=True)
        m.set_params({
            "coord_rotation__coordinates_names": ("x", "y"),
            "coord_rotation__number_axis": 2,
            "date_conversion__date_name": "transaction_date",
        })
        m.fit(
            X_train, y_train,
            model_features=features,
            X_val=X_val, y_val=y_val,
            verbose=False,
            early_stopping_rounds=5,
        )
        assert m.source_correction_terms == "Val"
        assert m.X_val is not None
        assert hasattr(m, "RMSE")
        assert hasattr(m, "smearing_factor")

    def test_fit_stores_model_features(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features)
        assert m.model_features == features


# ── predict ───────────────────────────────────────────────────────────────


class TestPredict:
    def test_predict_basic(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features)
        preds = m.predict(sample_df, verbose=False)
        assert len(preds) == len(sample_df)

    def test_predict_not_polars_raises(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features)
        with pytest.raises(AssertionError):
            m.predict(sample_df.to_pandas())

    def test_predict_miller_correction(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features, log_transform=True)
        raw = m.predict(sample_df, verbose=False)
        corrected = m.predict(
            sample_df,
            add_retransformation_correction=True,
            retransformation_method="Miller",
            verbose=False,
        )
        assert np.mean(corrected) > np.mean(raw)

    def test_predict_duan_correction(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features, log_transform=True)
        raw = m.predict(sample_df, verbose=False)
        corrected = m.predict(
            sample_df,
            add_retransformation_correction=True,
            retransformation_method="Duan",
            verbose=False,
        )
        assert np.mean(corrected) > np.mean(raw)

    def test_predict_invalid_method_raises(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features, log_transform=True)
        with pytest.raises(ValueError, match="retransformation_method"):
            m.predict(
                sample_df,
                add_retransformation_correction=True,
                retransformation_method="invalid",
                verbose=False,
            )

    def test_predict_no_correction_without_log(self, sample_df, target, features):
        m = _fit_model(sample_df, target, features, log_transform=False)
        raw = m.predict(sample_df, verbose=False)
        result = m.predict(
            sample_df,
            add_retransformation_correction=True,
            retransformation_method="Duan",
            verbose=False,
        )
        np.testing.assert_allclose(raw, result)


# ── assert_is_1d_array ───────────────────────────────────────────────────


class TestAssertIs1dArray:
    def test_assert_1d_none_passes(self):
        m = _make_model()
        m.assert_is_1d_array(None)  # should not raise

    def test_assert_1d_valid(self):
        m = _make_model()
        m.assert_is_1d_array(np.array([1, 2, 3]))

    def test_assert_1d_2d_raises(self):
        m = _make_model()
        with pytest.raises(AssertionError, match="1-dimensional"):
            m.assert_is_1d_array(np.array([[1, 2]]))

    def test_assert_1d_not_array_raises(self):
        m = _make_model()
        with pytest.raises(AssertionError, match="numpy array"):
            m.assert_is_1d_array([1, 2, 3])


# ── calibrate_model ──────────────────────────────────────────────────────


class TestCalibrateModel:
    def test_calibrate_not_twosteps_raises(self):
        with pytest.raises(AssertionError):
            calibrate_model(model="not a model")

    def test_calibrate_not_fitted_raises(self):
        m = _make_model()
        with pytest.raises(AssertionError, match="trained"):
            calibrate_model(model=m)

    def test_calibrate_converges(self, sample_df, target, features):
        n = len(sample_df)
        split = int(n * 0.7)
        m = _make_model(log_transform=True)
        m.set_params({
            "coord_rotation__coordinates_names": ("x", "y"),
            "coord_rotation__number_axis": 2,
            "date_conversion__date_name": "transaction_date",
        })
        m.fit(
            sample_df[:split], target[:split],
            model_features=features,
            X_val=sample_df[split:], y_val=target[split:],
            verbose=False,
            early_stopping_rounds=5,
        )
        model_out, converged = calibrate_model(
            model=m,
            calibration_variables=["region"],
            perform_distributional_calibration=True,
            convergence_rate=0.05,
            bounds=(0.5, 1.5),
            max_iter=200,
            verbose=False,
        )
        assert converged is True
        assert model_out.X_cal is not None

    def test_calibrate_bounds_respected(self, sample_df, target, features):
        n = len(sample_df)
        split = int(n * 0.7)
        m = _make_model(log_transform=True)
        m.set_params({
            "coord_rotation__coordinates_names": ("x", "y"),
            "coord_rotation__number_axis": 2,
            "date_conversion__date_name": "transaction_date",
        })
        m.fit(
            sample_df[:split], target[:split],
            model_features=features,
            X_val=sample_df[split:], y_val=target[split:],
            verbose=False,
            early_stopping_rounds=5,
        )
        bounds = (0.8, 1.2)
        model_out, _ = calibrate_model(
            model=m,
            calibration_variables=["region"],
            convergence_rate=0.05,
            bounds=bounds,
            max_iter=200,
            verbose=False,
        )
        ratios = model_out.X_cal["calibration_ratio_final"].to_numpy()
        assert ratios.min() >= bounds[0] - 1e-10
        assert ratios.max() <= bounds[1] + 1e-10


# ── train_calibration_model ──────────────────────────────────────────────


class TestTrainCalibrationModel:
    def test_train_cal_not_calibrated_raises(self):
        m = _make_model()
        with pytest.raises(AssertionError, match="Calibration"):
            train_calibration_model(model=m, verbose=False)

    def test_train_cal_basic(self, sample_df, target, features):
        n = len(sample_df)
        split = int(n * 0.7)
        m = _make_model(log_transform=True)
        m.set_params({
            "coord_rotation__coordinates_names": ("x", "y"),
            "coord_rotation__number_axis": 2,
            "date_conversion__date_name": "transaction_date",
        })
        m.fit(
            sample_df[:split], target[:split],
            model_features=features,
            X_val=sample_df[split:], y_val=target[split:],
            verbose=False,
            early_stopping_rounds=5,
        )
        m, converged = calibrate_model(
            model=m,
            calibration_variables=["region"],
            convergence_rate=0.05,
            max_iter=200,
            verbose=False,
        )
        assert converged
        m = train_calibration_model(
            model=m,
            calibration_model=lightgbm.LGBMRegressor(n_estimators=10, verbose=-1),
            verbose=False,
        )
        assert m.is_calibrated is True


# ── predict with calibration (integration) ───────────────────────────────


class TestPredictCalibration:
    def test_predict_calibration_after_full_pipeline(self, sample_df, target, features):
        n = len(sample_df)
        split = int(n * 0.7)
        m = _make_model(log_transform=True)
        m.set_params({
            "coord_rotation__coordinates_names": ("x", "y"),
            "coord_rotation__number_axis": 2,
            "date_conversion__date_name": "transaction_date",
        })
        m.fit(
            sample_df[:split], target[:split],
            model_features=features,
            X_val=sample_df[split:], y_val=target[split:],
            verbose=False,
            early_stopping_rounds=5,
        )
        m, converged = calibrate_model(
            model=m,
            calibration_variables=["region"],
            convergence_rate=0.05,
            max_iter=200,
            verbose=False,
        )
        assert converged
        m = train_calibration_model(
            model=m,
            calibration_model=lightgbm.LGBMRegressor(n_estimators=10, verbose=-1),
            verbose=False,
        )
        preds = m.predict(
            sample_df,
            add_retransformation_correction=True,
            retransformation_method="calibration",
            verbose=False,
        )
        assert len(preds) == len(sample_df)
        assert np.all(np.isfinite(preds))


# ── create_calibration_pipeline ──────────────────────────────────────────


class TestCreateCalibrationPipeline:
    def test_calibration_pipeline_has_3_steps(self):
        pipe = create_calibration_pipeline()
        assert len(pipe.steps) == 3
        names = [name for name, _ in pipe.steps]
        assert names == ["validate_features", "pandas_converter", "model"]


# ── r2_eval ──────────────────────────────────────────────────────────────


class TestR2Eval:
    def test_r2_eval_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        name, score, higher_better = r2_eval(y, y)
        assert name == "r2"
        assert score == pytest.approx(1.0)
        assert higher_better is True

    def test_r2_eval_with_lgbm_dataset(self):
        y = np.array([1.0, 2.0, 3.0])
        ds = lightgbm.Dataset(np.zeros((3, 1)), label=y)
        ds.construct()
        name, score, _ = r2_eval(y, ds)
        assert score == pytest.approx(1.0)
