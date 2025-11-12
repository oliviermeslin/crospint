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


# A custom transformer to validate the features entering the housing prices pipeline
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
    transaction_date_name (str): Name of the column with transaction dates.
    reference_date (str): Reference date in YYYY-MM-DD format. Defaults to "2010-01-01".
    """
    def __init__(self, transaction_date_name: str = None, reference_date: str = "2010-01-01"):

        # Check if the reference date is valid
        if not is_valid_ymd(reference_date):
            raise ValueError("The reference date is not valid. The format must be 'YYYY-MM-DD'.")

        self.transaction_date_name = transaction_date_name
        self.reference_date = reference_date
        self.is_fitted = False

    def set_params(self, transaction_date_name: str = None, reference_date: str = "2010-01-01"):
        """
        Set parameters for the transformer.

        Parameters:
        transaction_date_name (str): Name of the column with transaction dates.
        reference_date (str): Reference date in YYYY-MM-DD format.

        Returns:
        self
        """
        # Check if the reference date is valid
        if not is_valid_ymd(reference_date):
            raise ValueError("The reference date is not valid. The format must be 'YYYY-MM-DD'.")

        self.transaction_date_name = transaction_date_name
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
        if self.transaction_date_name not in X.columns:
            raise ValueError(f"Feature {self.transaction_date_name} is not in the data")

        # Raise an error if the transaction date is not a date
        if not isinstance(X[self.transaction_date_name].dtype, pl.Date):
            raise TypeError(f"Feature {self.transaction_date_name} is not of type date")

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
        if self.transaction_date_name not in X.columns:
            raise ValueError(f"Feature {self.transaction_date_name} is not in the data")

        # Raise an error if the transaction date is not a date
        if not isinstance(X[self.transaction_date_name].dtype, pl.Date):
            raise TypeError(f"Feature {self.transaction_date_name} is not of type date")

        # Calculate the number of days between each date and the reference date
        X = X.with_columns(
            (
                pl.col(self.transaction_date_name) - pl.Series([self.reference_date]).str.to_date()
            ).dt.total_days().alias(f"{self.transaction_date_name}")
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


# A custom transformer to validate the features entering the housing prices pipeline
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


def create_price_model_pipeline(
    model=lightgbm.LGBMRegressor(),
    presence_coordinates=True,
    convert_to_pandas_before_fit: bool = False
):
    """
    Create a pipeline for housing prices modelling

    Parameters:
    model (BaseEstimator, optional): Model to use for the housing prices modelling.
    Defaults to LGBMRegressor.

    Returns:
    Pipeline: The constructed pipeline.
    """
    if convert_to_pandas_before_fit:
        print("    Adding a step for Pandas conversion at the end of the preprocessing")

        if presence_coordinates:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("coord_rotation", AddCoordinatesRotation()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("pandas_converter", ConvertToPandas()),
                    ("price_model", model)
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("pandas_converter", ConvertToPandas()),
                    ("price_model", model)
                ]
            )
    else:
        if presence_coordinates:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("coord_rotation", AddCoordinatesRotation()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("price_model", model)
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ("validate_features", ValidateFeatures()),
                    ("date_conversion", ConvertDateToInteger()),
                    ("price_model", model)
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
        convert_to_pandas_before_fit=False,
        floor_area_name=None
    ):

        if price_sq_meter is True and floor_area_name is None:
            raise ValueError("The model uses price per square meter, \
but the name of the floor area variable is missing")

        self.log_transform = log_transform
        self.price_sq_meter = price_sq_meter
        self.feature_names_in = None
        self.is_price_model_fitted = False
        self.presence_coordinates = presence_coordinates
        self.convert_to_pandas_before_fit = convert_to_pandas_before_fit
        self.floor_area_name = floor_area_name
        self.calibration_table = None

        print("    Initiating an unfitted price prediction pipeline.")
        self.price_model_pipeline = create_price_model_pipeline(
            model=model,
            presence_coordinates=presence_coordinates,
            convert_to_pandas_before_fit=convert_to_pandas_before_fit
        )

        self.preprocessor = self.price_model_pipeline[:-1]
        self.model = self.price_model_pipeline[-1]

    # Pass parameters to pipeline
    def set_params(self, dico):
        self.price_model_pipeline.set_params(**dico)
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
        if "LGBMRegressor" in str(self.price_model_pipeline[-1].__class__):
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
            self.price_model_pipeline[-1].fit(
                X_transformed,
                y_transformed,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_names=eval_names,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                **kwargs
            )

        elif "RandomForestRegressor" in str(self.price_model_pipeline[-1].__class__):
            print("    Let's train a RandomForestRegressor!")
            print("    Training the model") if verbose else None
            self.price_model_pipeline[-1].fit(
                X_transformed,
                y_transformed,
                sample_weight=sample_weight
            )

        end_time = time.monotonic()

        print(f"    Training time of the price prediction model: {end_time - start_time} seconds") \
            if verbose else None

        if self.log_transform:
            # Compute the model's RMSE and the Duan 1983's correction term
            # (useful for the retransformation correction)
            print("    Compute the model's correction terms") if verbose else None
            if X_val is not None and y_val is not None:
                y_val_pred = self.price_model_pipeline.predict(X_val)
                self.RMSE = math.sqrt(metrics.mean_squared_error(y_val_transformed, y_val_pred))
                self.smearing_factor = np.mean(np.exp(y_val_transformed - y_val_pred))
                self.source_correction_terms = "Val"
            else:
                y_pred = self.price_model_pipeline.predict(X)
                self.RMSE = math.sqrt(metrics.mean_squared_error(y_transformed, y_pred))
                self.smearing_factor = np.mean(np.exp(y_transformed - y_pred))
                self.source_correction_terms = "Train"

            print("    RMSE = ", self.RMSE)
            print("    Smearing factor = ", self.smearing_factor)

        self.is_price_model_fitted = True

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

    def predict(
        self,
        X,
        iteration_range=None,
        add_retransformation_correction: bool = True,
        retransformation_method: str = "Duan",
        verbose: bool = True,
        **kwargs
    ):

        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        assert isinstance(add_retransformation_correction, bool), \
            "add_retransformation_correction must be True or False"

        if add_retransformation_correction and retransformation_method not in ["Duan", "Miller", "calibration"]:
            raise ValueError(
                "The retransformation_method argument must be either features 'Duan', 'Miller' or 'calibration'."
            )

        # Predict the local average
        print("    Predicting the target") if verbose else None
        y_pred = self.price_model_pipeline.predict(X)

        # Invert the target transformation
        print("    Invert the target transformation") if verbose \
            and (self.price_sq_meter or self.log_transform) else None
        y_pred = self.inverse_transform(X, y_pred)

        # Calibrate predictions if calibration is chosen
        if add_retransformation_correction and retransformation_method == "calibration":
            print("    The models includes a calibration step.") \

            # Calibrate the data
            df = (
                pl.DataFrame({"predicted_price": y_pred})
                .join_where(
                    self.calibration_table,
                    # .select("lower_limit", "upper_limit", "calibration_ratio", "quantile"),
                    (c.predicted_price >= c.lower_limit, c.predicted_price < c.upper_limit)
                )
                .with_columns(
                    calibrated_price=c.predicted_price * c.calibration_ratio
                )
            )

            if df.filter(c.calibration_ratio.is_null()).shape[0] > 0:
                raise ValueError("At least one observation could not be calibrated. Check the calibration table")

            y_pred = df["calibrated_price"].to_numpy().ravel()
            
            return y_pred

        if self.log_transform:
            if add_retransformation_correction:
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
            else:
                print("    There is no correction for the retransformation bias.") \
                    if verbose else None
        else:
            print("    The model has no log-transformation.") \
                if verbose else None

        return y_pred

def predict_market_value(
    X: pl.DataFrame,
    model,
    date_market_value=None,
    add_retransformation_correction=False,
    retransformation_method: str = "Duan",
    **kwargs
):
    """
    This function predicts market values.

    Parameters
    ----------
    X : pd.DataFrame or pl.DataFrame
        Feature matrix.

    date_market_value : None or a string in the '%Y-%m-%d' format
        The date at which market values must be predicted. See below.

    add_retransformation_correction : bool, default=True
        Whether to apply retransformation bias correction if `log_transform=True`.

    retransformation_method : {"Duan", "Miller"}, default="Duan"
        Method for retransformation correction.

    This function can be used in two ways: 
    - using the observed transaction date for each transaction (for instance in a test set);
      In this case `date_market_value` should be set to `None`, and the transaction data
      should be present in the features.
    - using a constant user-chosen date (for instance January, 1st).
      In this case `date_market_value` should be set to a date in the '%Y-%m-%d' format.

    Returns
    -------
    market_values : np.ndarray
        Predicted values in the original target scale.

    """
    # Extract the name of the feature containing the transaction name
    transaction_date_name = model.price_model_pipeline["date_conversion"].transaction_date_name

    if isinstance(X, pl.DataFrame):
        feature_names = X.columns
    elif isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()

    if date_market_value is not None and transaction_date_name in feature_names:
        raise ValueError(f"Data should not contain the column {transaction_date_name} \
            if date_market_value is not None.")

    if date_market_value is not None:
        print(f'    Predicting market values at date {date_market_value}.')
        X = (
            X
            .with_columns(
                pl.lit(date_market_value).str.to_date(format='%Y-%m-%d').alias(transaction_date_name),
                pl.lit(date_market_value[0:4]).str.to_integer().alias("anneemut"),
                pl.lit(date_market_value[5:7]).str.to_integer().alias("moismut")
            )
        )
    elif transaction_date_name in X.columns:
        print('    Predicting market values using transaction date from the data.')
    else:
        raise ValueError("The date for market value prediction is missing.")
    
    # Predict market values
    market_values = model.predict(X, **kwargs)

    return market_values
