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
import numpy as np
import polars as pl
from polars import col as c
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
import lightgbm
from datetime import datetime


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
    A custom transformer to transform a Polars DataFrame
    Parameters: None
    """
    def __init__(self):
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit the transformer by doing nothing

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """

        self.feature_names = X.columns
        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Convert to pandas

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pd.DataFrame.
        """
        return X.to_pandas()

    def fit_transform(self, X: pl.DataFrame, y=None):
        """
        Fit and transform the data in one step.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        pd.DataFrame: Same data converted to pandas.
        """
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
    presence_date=True,
    convert_to_pandas_before_fit: bool = False
):
    """
    Create a pipeline for spatio-temporal modelling

    Parameters:
    model (BaseEstimator, optional): Model to use as the last step of the pipeline.
    Defaults to LGBMRegressor.

    Returns:
    Pipeline: The constructed pipeline.
    """
    if convert_to_pandas_before_fit:
        print("    Adding a step for Pandas conversion at the end of the preprocessing")

        if presence_coordinates and presence_date:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("coord_rotation", AddCoordinatesRotation()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("pandas_converter", ConvertToPandas()),
                    ("model", model)
                ]
            )
        elif presence_coordinates:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("coord_rotation", AddCoordinatesRotation()),
                    ("pandas_converter", ConvertToPandas()),
                    ("model", model)
                ]
            )
        elif presence_date:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("pandas_converter", ConvertToPandas()),
                    ("model", model)
                ]
            )
    else:
        if presence_coordinates and presence_date:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("coord_rotation", AddCoordinatesRotation()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("model", model)
                ]
            )
        elif presence_coordinates:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("coord_rotation", AddCoordinatesRotation()),
                    ("model", model)
                ]
            )
        elif presence_date:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("model", model)
                ]
            )

    return pipe


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
        convert_to_pandas_before_fit=True,
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
        self.convert_to_pandas_before_fit = convert_to_pandas_before_fit
        self.floor_area_name = floor_area_name
        self.calibration_function = None
        self.y_calibration = None
        self.y_pred_calibration = None
        self.floor_area_calibration = None
        self.list_dates_calibration = None
        self.calibration_data = None

        print("    Initiating an unfitted price prediction pipeline.")
        self.pipe = create_model_pipeline(
            model=model,
            presence_coordinates=presence_coordinates,
            convert_to_pandas_before_fit=convert_to_pandas_before_fit
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
            # We need the prediction in level to build the calibration function
            # The prediction contains no retransformation correction
            self.y_pred_calibration = self.inverse_transform(X_val, y_pred)
            self.y_calibration = y_val
            self.floor_area_calibration = X_val[self.floor_area_name].to_numpy()
            if "date_conversion" in [name for name, _ in self.pipe.steps]:
                self.transaction_date_calibration = X_val[
                    self.pipe["date_conversion"].date_name
                ]
            self.source_correction_terms = "Val"
        else:
            y_pred = self.pipe.predict(X)
            y_true = y_transformed
            # We need the prediction in level to build the calibration function
            # The prediction contains no retransformation correction
            self.y_pred_calibration = self.inverse_transform(X, y_pred)
            self.y_calibration = y
            self.floor_area_calibration = X[self.floor_area_name].to_numpy()
            if "date_conversion" in [name for name, _ in self.pipe.steps]:
                self.transaction_date_calibration = X[
                    self.pipe["date_conversion"].date_name
                ]
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
        y=None,
        y_pred=None,
        floor_area=None,
        quantile_start: float = 0,
        quantile_end: float = 1
    ):

        self.assert_is_1d_array(y)
        self.assert_is_1d_array(y_pred)
        self.assert_is_1d_array(floor_area)
        if y is None or y_pred is None or floor_area is None:
            print(f"Calibrating the model using {self.source_correction_terms} data used \
in training")
            y = self.y_calibration
            y_pred = self.y_pred_calibration
            floor_area = self.floor_area_calibration

        self.list_dates_calibration = None
        self.calibration_data = None

        # Prepare data for calibration
        y = np.log(y / floor_area)
        y_pred = np.log(y_pred / floor_area)

        # Fit the calibration function on the whole distribution
        cal_func = IsotonicRegression(out_of_bounds="clip")
        cal_func.fit(
            np.sort(y_pred),
            np.sort(y)
        )

        if quantile_start > 0 or quantile_end < 1:
            print(f"    Restricting the calibration to the [{quantile_start}; {quantile_end}] range")
            lower_bound = cal_func.predict(np.quantile(y_pred, [quantile_start])).tolist()[0]
            upper_bound = cal_func.predict(np.quantile(y_pred, [quantile_end])).tolist()[0]
            cal_func = IsotonicRegression(
                out_of_bounds="clip",
                y_min=lower_bound,
                y_max=upper_bound
            )
            cal_func.fit(
                np.sort(y_pred),
                np.sort(y)
            )

        self.calibration_function = cal_func

        # Store calibration data
        self.y_pred_calibrated = self.floor_area_calibration \
            * np.exp(
                self.calibration_function.predict(
                    np.log(self.y_pred_calibration / self.floor_area_calibration)
                )
            )

        self.calibration_data = pl.DataFrame(
            {
                "y": self.y_calibration,
                "y_pred_calibration": self.y_pred_calibration,
                "y_pred_calibrated": self.y_pred_calibrated,
                "transaction_date": self.transaction_date_calibration
            }
        )

    def perform_time_calibration(
        self,
        list_dates_calibration: list = None
    ):
        if self.calibration_data is None:
            raise ValueError("The model must be calibrated before performing time calibration")
        if list_dates_calibration is None:
            raise ValueError("list_dates_calibration cannot be None")

        self.list_dates_calibration = np.unique(list_dates_calibration).tolist()

        df_time_intervals = (
            pl.DataFrame({"start": [None] + self.list_dates_calibration})
            .with_columns(
                c.start.str.strptime(pl.Date, format="%Y-%m-%d")
                .fill_null(pl.Series(["0001-01-01"]).str.strptime(pl.Date, format="%Y-%m-%d"))
            )
            .sort("start")
            .with_columns(
                end=c.start.shift(-1)
                .fill_null(pl.Series(["3000-01-01"]).str.strptime(pl.Date, format="%Y-%m-%d"))
            )
        )

        calibration_data_ranges = (
            self.calibration_data
            .join_where(
                df_time_intervals,
                pl.col("transaction_date") >= pl.col("start"),
                pl.col("transaction_date") < pl.col("end")
            )
        )

        self.time_calibration_data = (
            calibration_data_ranges
            .group_by("start", "end")
            .agg(
                total_obs=c.y.sum(),
                total_pred=c.y_pred_calibrated.sum()
            )
            .with_columns(
                ratio=c.total_obs/c.total_pred
            )
            .sort("start")
        )

    def predict(
        self,
        X,
        iteration_range=None,
        add_retransformation_correction: bool = False,
        retransformation_method: str = None,
        apply_time_calibration: bool = False,
        verbose: bool = True,
        **kwargs
    ):

        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        assert isinstance(add_retransformation_correction, bool), \
            "add_retransformation_correction must be True or False"

        if add_retransformation_correction and retransformation_method not in [None, "Duan", "Miller", "calibration"]:
            raise ValueError(
                "The retransformation_method argument must be either None, 'Duan', 'Miller' or 'calibration'."
            )

        # Predict the local average
        print("    Predicting the target") if verbose else None
        y_pred = self.pipe.predict(X)

        # Invert the target transformation
        print("    Invert the target transformation") if verbose \
            and (self.price_sq_meter or self.log_transform) else None
        y_pred = self.inverse_transform(X, y_pred)

        # Correct raw predictions
        if add_retransformation_correction:
            # Calibrate predictions if calibration is chosen
            if retransformation_method == "calibration":
                print("    The models includes a calibration step.")

                # Calibrate the predictions
                y_pred = (
                    X[self.floor_area_name].to_numpy() \
                    # Compute calibrated price_sqm in level
                    * np.exp(
                        # Calibrate this raw prediction
                        self.calibration_function.predict(
                            # Start from raw pipeline prediction (log_price_sqm)
                            np.log(y_pred / X[self.floor_area_name].to_numpy())
                        )
                    )
                )

                if apply_time_calibration:
                    df_time_calibration = (
                        X
                        .select(self.pipe["date_conversion"].date_name)
                        # join_where does not keep row order, so we need a row number to put
                        # final predictions in the right order
                        .with_row_count(name="row_identifier", offset=0)
                        .with_columns(pl.Series(y_pred).alias("y_pred"))
                        .join_where(
                            self.time_calibration_data,
                            pl.col(self.pipe["date_conversion"].date_name) >= pl.col("start"),
                            pl.col(self.pipe["date_conversion"].date_name) < pl.col("end")
                        )
                        .with_columns(y_pred_calibrated=c.y_pred_calibrated * c.ratio)
                        .sort("row_identifier")
                        .drop("row_identifier")
                    )
                    if X.shape[0] != df_time_calibration.shape[0]:
                        raise ValueError("    There are duplicates in the time calibration step")
                    if df_time_calibration.filter(c.ratio.is_null()).shape[0] > 0:
                        raise ValueError("    There are missing values in the time calibration step")
                    return df_time_calibration["y_pred_calibrated"].to_numpy()
                else:
                    return y_pred

            if self.log_transform:
                print("    The models includes a correction of the retransformation bias.") \
                    if verbose else None
                # Use the Duan's 1983 smearing factor correction
                if retransformation_method == "Duan":
                    print("    The prediction is corrected using Duan's 1983 smearing factor.") \
                        if verbose else None
                    global_correction = self.smearing_factor
                # Use the Miller's 1984 retransformation correction
                elif retransformation_method == "Miller":
                    print("    The prediction is corrected using Miller's 1984 method.") \
                        if verbose else None
                    global_correction = np.exp((self.RMSE ** 2) / 2)
                print("    Average correction = ", round(100 * (global_correction - 1), 2), '%')

                # Apply the correction to the prediction
                y_pred = y_pred * global_correction

                return y_pred
            else:
                print("    The model has no log-transformation.") \
                    if verbose else None
                return y_pred

        else:
            return y_pred
