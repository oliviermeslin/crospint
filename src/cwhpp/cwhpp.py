# MIT License

# Copyright (c) 2025 Olivier Meslin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import time
import math
from datetime import datetime
import copy
import numpy as np
import pandas as pd
import polars as pl
from polars import col as c
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from sklearn import metrics
import lightgbm
from lightgbm.callback import EarlyStopException


def rotate_point(x, y, angle, center=None):
    """
    Rotate a 2D point counterclockwise by a given angle (in degrees) around a given center.

    Parameters:
    x (float or np.array): x-coordinate(s) to rotate.
    y (float or np.array): y-coordinate(s) to rotate.
    angle (float): Angle in degrees by which to rotate the point.
    center (tuple, optional): Center of rotation as (cx, cy). Defaults to the origin (0, 0).

    Returns:
    tuple: Rotated x and y coordinates.
    """
    cx, cy = center if center else (0, 0)
    dx = x - cx
    dy = y - cy
    ang_rad = math.radians(angle)

    xx = cx + dx * math.cos(ang_rad) - dy * math.sin(ang_rad)
    yy = cy + dx * math.sin(ang_rad) + dy * math.cos(ang_rad)

    return xx, yy


# A custom transformer to validate the features entering the pipeline
class ValidateFeatures(BaseEstimator, TransformerMixin):
    """
    A custom transformer to validate features
    This transformer checks that all model features are present in the data
    and returns a dataframe with the right features in the right order
    Parameters: None
    """
    def __init__(self):
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit the transformer by checking for valid coordinate names and calculating the mean center.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        self.feature_names = X.columns
        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Validate the features

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pl.DataFrame: dataframe with the right features in the right order.
        """

        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        features_X = X.columns

        missing_features = []
        for var in self.feature_names:
            if var not in features_X:
                missing_features += [var]
                print(f'Feature {var} is missing in the data')

        if len(missing_features) > 0:
            raise ValueError("Some features are missing in the data.")

        return X[self.feature_names]

    def fit_transform(self, X: pl.DataFrame, y=None):
        """
        Fit and transform the data in one step.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        pl.DataFrame: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self):
        """
        Get the names of the transformed features.

        Returns:
        list: Names of the transformed features.
        """
        return self.feature_names


# A custom transformer to rotate geographical coordinates an arbitrary number of times
class AddCoordinatesRotation(BaseEstimator, TransformerMixin):
    """
    A custom transformer to rotate geographical coordinates an arbitrary number of times.

    Parameters:
    coordinates_names (tuple): Tuple of column names representing the x and y coordinates.
    number_axis (int): The number of rotations to apply to the coordinates.
    """
    def __init__(self, coordinates_names: tuple = None, number_axis: int = None):
        self.coordinates_names = coordinates_names
        self.number_axis = number_axis
        self.is_fitted = False

    def set_params(self, coordinates_names: tuple = None, number_axis: int = None):
        """
        Set parameters for the transformer.

        Parameters:
        coordinates_names (tuple): Tuple of column names representing the x and y coordinates.
        number_axis (int): The number of rotations to apply to the coordinates.

        Returns:
        self
        """
        self.coordinates_names = coordinates_names
        self.rotated_coordinates_names = []
        self.number_axis = number_axis
        return self

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit the transformer by checking for valid coordinate names and calculating the mean center.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        coordinates_names = self.coordinates_names

        # Raise an error if the coordinates are not correct
        if coordinates_names is None:
            raise ValueError("Argument coordinates_names is missing.")
        if len(coordinates_names) != 2:
            raise ValueError("There must be exactly two coordinates.")
        if coordinates_names[0] not in X.columns:
            raise ValueError(f"Coordinate {coordinates_names[0]} is not in the data.")
        if coordinates_names[1] not in X.columns:
            raise ValueError(f"Coordinate {coordinates_names[1]} is not in the data.")

        # Raise an error if the number of axis is missing
        if self.number_axis is None:
            raise ValueError("Argument number_axis is missing.")

        x_coord, y_coord = self.coordinates_names
        # Compute the mean coordinates of the data
        self.center = (X[x_coord].mean(), X[y_coord].mean())

        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Rotate the coordinates and return the modified data.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pl.DataFrame: Transformed data with additional rotated coordinates.
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        x_coord, y_coord = self.coordinates_names
        rotated_coordinates_names = [x_coord, y_coord]

        for i in range(1, self.number_axis):
            # Compute coordinates after rotation
            x_temp, y_temp = rotate_point(
                x=X[x_coord],
                y=X[y_coord],
                angle=360 * (i / self.number_axis),
                center=self.center
            )

            # Add the rotated coordinates to the data
            X = X.with_columns(
                [
                    x_temp.alias(f"{x_coord}_rotated{i}"),
                    y_temp.alias(f"{y_coord}_rotated{i}")
                ]
            )
            rotated_coordinates_names = rotated_coordinates_names + [
                f"{x_coord}_rotated{i}", f"{y_coord}_rotated{i}"
            ]

        self.rotated_coordinates_names = rotated_coordinates_names
        self.names_features_output = X.columns
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in one step.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        pl.DataFrame: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self):
        """
        Get the names of the transformed features.

        Returns:
        list: Names of the transformed features.
        """
        return self.names_features_output


def is_valid_ymd(date_str: [str, list]):
    """
    Validate whether a date or a list of dates are string in the 'YYYY-MM-DD' format

    Returns:
    list: Names of the transformed features.
    """
    if isinstance(date_str, str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return False
    elif isinstance(date_str, list):
        for date in date_str:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return False
    return True


# A custom transformer to convert a date variable to a numerical variable
class ConvertDateToInteger(BaseEstimator, TransformerMixin):
    """
    A custom transformer to convert transaction dates to integers (days since a reference date).

    Parameters:
    date (str): Name of the column with transaction dates.
    reference_date (str): Reference date in YYYY-MM-DD format. Defaults to "2010-01-01".
    """
    def __init__(self, date_name: str = None, reference_date: str = "2010-01-01"):

        # Check if the reference date is valid
        if not is_valid_ymd(reference_date):
            raise ValueError("The reference date is not valid. The format must be 'YYYY-MM-DD'.")

        self.date_name = date_name
        self.reference_date = reference_date
        self.is_fitted = False

    def set_params(self, date_name: str = None, reference_date: str = "2010-01-01"):
        """
        Set parameters for the transformer.

        Parameters:
        date_name (str): Name of the column with transaction dates.
        reference_date (str): Reference date in YYYY-MM-DD format.

        Returns:
        self
        """
        # Check if the reference date is valid
        if not is_valid_ymd(reference_date):
            raise ValueError("The reference date is not valid. The format must be 'YYYY-MM-DD'.")

        self.date_name = date_name
        self.reference_date = reference_date
        self.names_features_output = None
        return self

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit the transformer by validating the transaction date column.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"

        # Raise an error if the transaction date is not in the data
        if self.date_name not in X.columns:
            raise ValueError(f"Feature {self.date_name} is not in the data")

        # Raise an error if the transaction date is not a date
        if not isinstance(X[self.date_name].dtype, pl.Date):
            raise TypeError(f"Feature {self.date_name} is not of type date")

        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Convert the transaction date to an integer representing the number of days
        since the reference date.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pl.DataFrame: Transformed data with integer representation of dates.
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"

        # Raise an error if the transaction date is not in the data
        if self.date_name not in X.columns:
            raise ValueError(f"Feature {self.date_name} is not in the data")

        # Raise an error if the transaction date is not a date
        if not isinstance(X[self.date_name].dtype, pl.Date):
            raise TypeError(f"Feature {self.date_name} is not of type date")

        # Calculate the number of days between each date and the reference date
        X = X.with_columns(
            (
                pl.col(self.date_name) - pl.Series([self.reference_date]).str.to_date()
            ).dt.total_days().alias(f"{self.date_name}")
        )

        # Store feature names
        self.names_features_output = X.columns

        return X

    def fit_transform(self, X: pl.DataFrame, y=None):

        # Fit the transformer
        self.fit(X, y)

        # Transform the data
        X = self.transform(X, y)

        return X

    def get_feature_names_out(self):
        """
        Get the names of the transformed features.

        Returns:
        list: Names of the transformed features.
        """
        return self.names_features_output


# A custom transformer to convert a polars DataFrame into Pandas
class ConvertToPandas(BaseEstimator, TransformerMixin):
    """
    Convert a Polars DataFrame to Pandas while:
    - converting string columns to categorical in Polars
    - storing category -> integer mappings at fit time
    - reapplying the same encoding at transform time
    """

    def __init__(self):
        self.feature_names = None
        self.string_cols = None
        self.category_mappings = {}
        self.is_fitted = False

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit by detecting string columns, converting them to categorical,
        and storing category -> integer mappings.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        self.feature_names = X.columns
        self.category_mappings = {}

        # Detect string columns
        self.string_cols = [
            col for col, dtype in zip(X.columns, X.dtypes)
            if dtype in [pl.Utf8]
        ]

        if len(self.string_cols) > 0:
            # Extract categories
            for col in self.string_cols:
                # Replace null values with a non-null value
                X = (
                    X
                    .with_columns(pl.col(col).fill_null("missing").alias(col))
                )
                self.category_mappings[col] = sorted(X[col].unique().to_numpy().tolist())

        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Apply stored categorical encodings and convert to Pandas.
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer has not been fitted")

        if not isinstance(X, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        for col in self.string_cols:
            # Replace null values in categorical features with a non-null value
            X = (
                    X
                    .with_columns(pl.col(col).fill_null("missing").alias(col))
                )

        df = X.to_pandas()
        # Convert all string variables to categorical with the same encoding
        for col in self.string_cols:
            df[col] = pd.Categorical(
                df[col],
                categories=self.category_mappings[col]
            )

        # Convert to pandas (all numeric / safe types now)
        return df

    def fit_transform(self, X: pl.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self):
        """

        Returns:
        list: Names of the features.
        """
        return self.feature_names


def create_model_pipeline(
    model=lightgbm.LGBMRegressor(),
    presence_coordinates=True,
    presence_date=True
):
    """
    Create a pipeline for spatio-temporal modelling

    Parameters:
    model (BaseEstimator, optional): Model to use as the last step of the pipeline.
    Defaults to LGBMRegressor.

    Returns:
    Pipeline: The constructed pipeline.
    """

    # Start by validating the features
    steps = [
        ("validate_features", ValidateFeatures())
    ]

    if presence_coordinates:
        steps.append(
            ("coord_rotation", AddCoordinatesRotation())
        )

    if presence_date:
        steps.append(
            ("date_conversion", ConvertDateToInteger())
        )

    # Add a conversion to Pandas
    steps.append(
        ("pandas_converter", ConvertToPandas())
    )

    # Add the model
    steps.append(
        ("model", model)
    )

    # Create the pipeline
    pipe = Pipeline(steps)

    return pipe


def create_calibration_pipeline(
    model=lightgbm.LGBMRegressor()
):

    steps = [
        ("validate_features", ValidateFeatures()),
        ("pandas_converter", ConvertToPandas())
    ]

    # Add the model
    steps.append(("model", model))
    pipe = Pipeline(
        steps=steps
    )
    return pipe


def compute_calibration_ratios(
    X: pl.DataFrame = None,
    calibration_variables: list = None,
    perform_distributional_calibration: bool = True,
    raw_prediction_variable="predicted_price",
    cal_prediction_variable="predicted_price_cal",
    target_variable="target",
    bounds=(0.5, 1.5)
):

    assert perform_distributional_calibration or calibration_variables is not None, \
        "At least one form of calibration should be performed"

    lower_bound, upper_bound = bounds

    if perform_distributional_calibration:
        # Train an isotonic regression mapping the distribution of predicted prices
        # to the distribution of observed prices
        cal_func = IsotonicRegression(out_of_bounds="clip")
        cal_func.fit(
            np.sort(X[cal_prediction_variable].to_numpy()),
            np.sort(X[target_variable].to_numpy())
        )

        # Use the isotonic regression to compute raw calibration ratios
        calibration_ratio_isotonic = (
            cal_func.predict(X[cal_prediction_variable].to_numpy())
            / X[cal_prediction_variable].to_numpy()
        )

        # Perform distributional calibration
        X = (
            X
            # Update the calibration ratio
            .with_columns(
                calibration_ratio=(
                    (c.calibration_ratio*pl.Series(calibration_ratio_isotonic))
                    .clip(lower_bound=lower_bound, upper_bound=upper_bound)
                )
            )
            # Adjust raw predictions
            .with_columns(
                (pl.col(raw_prediction_variable)*c.calibration_ratio).alias(cal_prediction_variable)
            )
        )

    if calibration_variables is not None:
        # Adjust the predictions to match the marginal distribution of each calibration variable
        for calibration_variable in calibration_variables:
            X = (
                X
                # Compute marginal distributions
                .with_columns(
                    total_pred_temp=(
                        pl.col(cal_prediction_variable).sum().over(calibration_variable)
                    ),
                    total_obs_temp=(
                        pl.col(target_variable).sum().over(calibration_variable)
                    ),
                )
                .with_columns(
                    # Update the calibration ratio
                    calibration_ratio=(
                        (c.calibration_ratio*(c.total_obs_temp / c.total_pred_temp))
                        .clip(lower_bound=lower_bound, upper_bound=upper_bound)
                    )
                )
                # Adjust raw predictions
                .with_columns(
                    (
                        pl.col(raw_prediction_variable)*c.calibration_ratio
                    ).alias(cal_prediction_variable)
                )
            )
    return X


# Define a custom early stopping callback based on r2
def stop_on_train_r2(threshold=None):
    def _callback(env):
        # Find training dataset
        for data_name, eval_name, value, _ in env.evaluation_result_list:
            if data_name == "Train" and eval_name == "r2":
                if value >= threshold:
                    print(
                            f"[Early stopping] "
                            f"Iteration {env.iteration + 1}, "
                            f"training {eval_name} = {value:.6f} "
                            f"(threshold = {threshold})"
                        )
                    raise EarlyStopException(
                        best_iteration=env.iteration,
                        best_score=env.evaluation_result_list
                    )
    return _callback


# Define a custom evaluation metric
def r2_eval(preds, dataset):
    if isinstance(dataset, lightgbm.Dataset):
        y_true = dataset.get_label()
    else:
        # sklearn API
        y_true = dataset
    return "r2", r2_score(y_true, preds), True


class TwoStepsModel(BaseEstimator):
    """
    A custom estimator that combines two steps: transformation of the target
    and housing price modelling.
    """
    def __init__(
        self,
        model=lightgbm.LGBMRegressor(),
        log_transform=None,
        price_sq_meter=None,
        presence_coordinates=True,
        presence_date=True,
        floor_area_name=None
    ):

        if price_sq_meter is True and floor_area_name is None:
            raise ValueError("The model uses price per square meter, \
but the name of the floor area variable is missing")

        self.log_transform = log_transform
        self.price_sq_meter = price_sq_meter
        self.feature_names_in = None
        self.is_model_fitted = False
        self.presence_coordinates = presence_coordinates
        self.presence_date = presence_date
        self.floor_area_name = floor_area_name
        self.X_val = None
        self.y_val = None

        print("    Initiating an unfitted price prediction pipeline.")
        self.pipe = create_model_pipeline(
            model=model,
            presence_coordinates=presence_coordinates,
            presence_date=presence_date
        )

        self.preprocessor = self.pipe[:-1]
        self.model = self.pipe[-1]

    # Pass parameters to pipeline
    def set_params(self, dico):
        self.pipe.set_params(**dico)
        return self

    def fit(
        self,
        X: pl.DataFrame,
        y,
        model_features=None,
        X_val=None,
        y_val=None,
        sample_weight=None,
        sample_weight_val=None,
        verbose=True,
        log_evaluation_period=100,
        early_stopping_rounds=25,
        **kwargs
    ):

        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        if X_val is not None:
            assert isinstance(X_val, pl.DataFrame), "X_val must be a Polars DataFrame"

        # Store feature names
        if model_features is not None:
            self.model_features = model_features
        else:
            self.model_features = X.columns

        print(f"    Average value of y before transformation: {np.mean(y)}") \
            if verbose else None

        # Transform the target according to the settings
        print("    Transforming the target") if verbose \
            and (self.log_transform or self.price_sq_meter) else None
        y_transformed = self.transform(X, y)

        print(f"    Average value of y after transformation: {np.mean(y_transformed)}") \
            if verbose else None

        # Fit the preprocessor
        print("    Fit the preprocessor") if verbose else None
        self.preprocessor.fit(X.select(model_features), y)

        # Transform the data with the preprocessor
        print("    Transform training data with the preprocessor") if verbose else None
        X_transformed = self.preprocessor.transform(X.select(model_features))

        # Prepare validation data is any
        if X_val is not None and y_val is not None:
            print("    Transform validation data") if verbose else None
            X_val_transformed = self.preprocessor.transform(X_val.select(model_features))
            y_val_transformed = self.transform(X_val, y_val)

            eval_set = [
                (X_transformed, y_transformed),
                (X_val_transformed, y_val_transformed)
            ]
            eval_names = ["Train", "Validation"]

        else:
            eval_set = [
                (X_transformed, y_transformed)
            ]
            eval_names = ["Train"]

        start_time = time.monotonic()
        if "LGBMRegressor" in str(self.pipe[-1].__class__):
            print("    Let's train a LGBMRegressor!")
            callbacks = [
                lightgbm.log_evaluation(period=log_evaluation_period),
                lightgbm.early_stopping(stopping_rounds=early_stopping_rounds)
            ]

            eval_weights = (sample_weight, sample_weight_val)
            eval_sample_weight = [].extend([obj for obj in eval_weights if obj is not None])
            if eval_sample_weight == []:
                eval_sample_weight = None

            print("    Training the model") if verbose else None
            self.pipe[-1].fit(
                X_transformed,
                y_transformed,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_names=eval_names,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                **kwargs
            )

        elif "RandomForestRegressor" in str(self.pipe[-1].__class__):
            print("    Let's train a RandomForestRegressor!")
            print("    Training the model") if verbose else None
            self.pipe[-1].fit(
                X_transformed,
                y_transformed,
                sample_weight=sample_weight
            )

        end_time = time.monotonic()

        print(f"    Training time of the price prediction model: {end_time - start_time} seconds") \
            if verbose else None

        print("    Fit correction terms") if verbose else None
        if X_val is not None and y_val is not None:
            y_pred = self.pipe.predict(X_val)
            y_true = y_val_transformed
            self.source_correction_terms = "Val"
            self.X_val = X_val
            self.y_val = y_val
        else:
            y_pred = self.pipe.predict(X)
            y_true = y_transformed
            self.source_correction_terms = "Train"

        if self.log_transform:
            # Compute the model's RMSE and the Duan 1983's correction term
            # (useful for the retransformation correction)
            print("    Compute the model's correction terms") if verbose else None
            self.RMSE = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
            self.smearing_factor = np.mean(np.exp(y_true - y_pred))

            print("    RMSE = ", self.RMSE)
            print("    Smearing factor = ", self.smearing_factor)

        self.is_model_fitted = True

        return self

    def transform(self, X, y):

        # Compute the price per square meter
        if self.price_sq_meter:
            y = y / X[self.floor_area_name].to_numpy()
        # Take the logarithm
        if self.log_transform:
            y = np.log(y)
        return y

    def inverse_transform(self, X, y):
        """
        Invert the target transformation according to the model specification.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            Feature matrix, used if `price_sq_meter=True`.

        y : array-like
            Transformed predictions to invert.

        Returns
        -------
        y_original : np.ndarray
            Predictions in the original target scale.
        """
        # Take the exponential if the model uses log
        if self.log_transform:
            y = np.exp(y)
        # Multiply by the floor area if the model uses the price per square meter
        if self.price_sq_meter:
            y = y * X[self.floor_area_name].to_numpy()

        return y

    def assert_is_1d_array(
        self,
        obj
    ):
        if obj is not None:
            assert isinstance(obj, np.ndarray), "Object is not a numpy array"
            assert obj.ndim == 1, "Array is not 1-dimensional"

    def calibrate_model(
        self,
        X: pl.DataFrame = None,
        y=None,
        calibration_variables: list = None,
        perform_distributional_calibration: bool = True,
        convergence_rate: float = 1e-3,
        bounds: tuple = (0.5, 1.5),
        max_iter: int = 100,
        calibration_model=lightgbm.LGBMRegressor(
            n_estimators=100,
            num_leaves=1023,
            max_depth=12,
            learning_rate=0.5,
            min_child_samples=20,
            max_bins=10000,
            random_state=123456,
            verbose=-1
        ),
        bounds: tuple = (0.5, 1.5),
        verbose: bool = True
    ):

        assert self.is_model_fitted, "The model must be trained before being calibrated"

        if (X is None or y is None):
            if self.X_val is not None and self.y_val is not None:
                print("    Using the validation dataset as calibration set.") if verbose else None
                X = copy.deepcopy(self.X_val)
                y = copy.deepcopy(self.y_val)
        else:
            raise ValueError("    Calibration data is missing.")

        self.assert_is_1d_array(y)
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        assert X.shape[0] == len(y), "y and X must have the same length"
        assert isinstance(calibration_variables, list) or calibration_variables is None, \
            "calibration_variables must be a list"
        assert calibration_variables is not None or perform_distributional_calibration, \
            "calibration_variables cannot be None if there is no distributional calibration"

        # Check that X contains all necessary data
        missing_vars = []
        for var in calibration_variables + self.model_features:
            if var not in X.columns:
                missing_vars.extend(var)
                print(f"Variable {var} is missing in the calibration set.")
        if len(missing_vars) > 0:
            raise ValueError("Some variables are missing in the calibration set.")

        assert len(bounds) == 2, 'Bounds must be a tuple of length 2'

        # Step 1: Predicting prices on the calibration set
        print("    Predicting prices on the calibration set") if verbose else None
        raw_predictions = self.predict(
            X,
            add_retransformation_correction=False,
            retransformation_method=None,
            verbose=False
        )
        # Add the prediction to the calibration set
        X_cal = X.with_columns(
            predicted_price=pl.Series(raw_predictions),
            target=pl.Series(y)
        ).with_columns(
            predicted_price_sqm=c.predicted_price/pl.col(self.floor_area_name)
        )

        # Step 2: Performing the calibration
        print("    Performing the calibration") if verbose else None

        # Initialize the calibrated price
        X_cal = X_cal.with_columns(
            predicted_price_cal=c.predicted_price,
            calibration_ratio=pl.lit(1)
        )

        if calibration_variables is None:
            print("    There are no calibration variables. \
Only distributional calibration will be performed.") if verbose else None

        if perform_distributional_calibration is False:
            print("    There is no distributional calibration. \
Only marginal calibration will be performed.") if verbose else None

        # Perform the iterative calibration
        nb_iter = 0
        max_conv = 10 * convergence_rate
        while max_conv > convergence_rate:
            if nb_iter >= max_iter:
                raise RuntimeError(f"    Algorithm failed to converge after {max_iter} iterations. \
You may try again with looser bounds, higher convergence thresholds or less calibration variables.")

            X_cal = compute_calibration_ratios(
                X=X_cal,
                calibration_variables=calibration_variables,
                perform_distributional_calibration=perform_distributional_calibration,
                raw_prediction_variable="predicted_price",
                cal_prediction_variable="predicted_price_cal",
                target_variable="target",
                bounds=bounds
            )

            # Check if convergence has been reached
            if calibration_variables is not None:

                max_conv = 0
                for var in calibration_variables:
                    largest_gap = (
                        X_cal
                        .group_by(var)
                        .agg(
                            nb=pl.len(),
                            ratio=(c.predicted_price_cal.sum()/c.target.sum()-1).abs()
                        )
                        # Remove very small categories
                        .filter(c.nb/c.nb.sum() > 0.01)
                        .select(c.ratio.max())
                        .item()
                    )
                    if largest_gap > max_conv:
                        max_conv = largest_gap
            nb_iter += 1

        print(f"    The calibration procedure converged after {nb_iter} iterations") \
            if verbose else None
        print(f"    max_conv = {max_conv}") if verbose else None

        # Compute final calibration ratios
        X_cal = X_cal.with_columns(
            calibration_ratio_final=c.predicted_price_cal/c.predicted_price
        )

        # Step 3: Train the calibration model
        print("    Training the calibration model") if verbose else None
        self.calibration_model = create_calibration_pipeline(
            model=calibration_model
        )

        # Train the model
        # This model is intentionally overfit
        self.calibration_model.fit(
            X_cal.select(calibration_variables + ["predicted_price"]),
            X_cal["calibration_ratio_final"].to_numpy()
        )
        # Predict calibration ratios on the calibration set
        predicted_ratios = self.calibration_model.predict(X_cal)

        # Add predicted calibration ratios in the calibration set
        self.calibration_variables = calibration_variables
        self.X_cal = (
            X_cal
            .select(
                calibration_variables + [
                    "target",
                    "predicted_price",
                    "calibration_ratio_final",
                    "predicted_price_cal"
                ]
            )
            .with_columns(
                calibration_ratio_final_pred=pl.Series(predicted_ratios)
            )
            .with_columns(
                predicted_price_cal_pred=c.predicted_price*c.calibration_ratio_final_pred
            )
        )
        self.is_calibrated = True

    def calibrate_prediction(
        self,
        X,
        y
    ):

        # Add the predictions to the dataframe
        X = (
            X
            .with_columns(predicted_price=pl.Series(y))
            .select(
                self.calibration_variables + ["predicted_price"]
            )
        )

        # Predict calibration ratios
        predicted_calibration_ratios = self.calibration_model.predict(X)

        # Compute calibrated predictions by applying calibration ratios
        y_calibrated = y * predicted_calibration_ratios

        return y_calibrated

    def predict(
        self,
        X,
        iteration_range=None,
        add_retransformation_correction: bool = False,
        retransformation_method: str = None,
        verbose: bool = True,
        **kwargs
    ):

        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        assert isinstance(add_retransformation_correction, bool), \
            "add_retransformation_correction must be True or False"

        if add_retransformation_correction and \
                retransformation_method not in [None, "Duan", "Miller", "calibration"]:
            raise ValueError(
                "retransformation_method must be either None, 'Duan', 'Miller' or 'calibration'."
            )

        # Check that calibration variables are present in the data
        if retransformation_method == "calibration":
            assert self.is_calibrated, "The model is not calibrated"
            missing_calvar = []
            for var in self.calibration_variables:
                if var not in X.columns:
                    missing_calvar.extend(var)
                    print(f"Variable {var} is missing in the calibration set.")
            if len(missing_calvar) > 0:
                raise ValueError("Some calibration variables are missing in the data.")

        # Predict the price
        print("    Predicting the target") if verbose else None
        y_pred = self.pipe.predict(X)

        # Invert the target transformation
        print("    Invert the target transformation") if verbose \
            and (self.price_sq_meter or self.log_transform) else None
        y_pred = self.inverse_transform(X, y_pred)

        # Return raw predictions if no correction is applied
        if not add_retransformation_correction or retransformation_method is None:
            print("    No retransformation correction is applied to raw predictions.") \
                if verbose else None
            return y_pred

        else:
            # Calibrate predictions if calibration is chosen
            if retransformation_method == "calibration":
                print("    Raw predictions are calibrated.")

                # Calibrate the prediction
                return self.calibrate_prediction(X, y_pred)

            # Apply retransformation correction if the model includes log
            elif self.log_transform:
                print("    Final predictions include a correction of the retransformation bias.") \
                    if verbose else None
                # Use the Duan's 1983 smearing factor correction
                if retransformation_method == "Duan":
                    print("    Raw predictions are corrected using Duan's 1983 smearing factor.") \
                        if verbose else None
                    global_correction = self.smearing_factor
                # Use the Miller's 1984 retransformation correction
                elif retransformation_method == "Miller":
                    print("    Raw predictions are corrected using Miller's 1984 method.") \
                        if verbose else None
                    global_correction = np.exp((self.RMSE ** 2) / 2)
                print("    Average correction = ", round(100 * (global_correction - 1), 2), '%')

                # Apply the correction to the prediction
                return y_pred * global_correction
            else:
                print("    Raw predictions are not corrected because\
the model has no log-transformation.") if verbose else None
                return y_pred
