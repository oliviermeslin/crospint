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


import math
from datetime import datetime
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import lightgbm


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
    A custom transformer to convert dates to integers (days since a reference date).

    Parameters:
    date (str): Name of the column with dates.
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
        date_name (str): Name of the column with dates.
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
        Fit the transformer by validating the date column.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"

        # Raise an error if the date is not in the data
        if self.date_name not in X.columns:
            raise ValueError(f"Feature {self.date_name} is not in the data")

        # Raise an error if the date is not a date
        if not isinstance(X[self.date_name].dtype, pl.Date):
            raise TypeError(f"Feature {self.date_name} is not of type date")

        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Convert the date to an integer representing the number of days elapsed
        since the reference date.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pl.DataFrame: Transformed data with integer representation of dates.
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"

        # Raise an error if the date is not in the data
        if self.date_name not in X.columns:
            raise ValueError(f"Feature {self.date_name} is not in the data")

        # Raise an error if the date is not a date
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

        df = X.to_pandas()
        # Convert all string variables to categorical with the same encoding
        for col in self.string_cols:
            df[col] = pd.Categorical(
                df[col],
                categories=self.category_mappings[col],
                ordered=False
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

    # Add a coordinates rotation step if chosen
    if presence_coordinates:
        steps.append(
            ("coord_rotation", AddCoordinatesRotation())
        )

    # Add a date conversion step if chosen
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
