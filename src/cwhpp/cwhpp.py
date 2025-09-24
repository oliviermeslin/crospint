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
import copy
import math
import numpy as np
import pandas as pd
import polars as pl
from polars import col as c
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import metrics
import lightgbm

############################################################
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
############################################################

# Rerunning with the PatchedRegressor fixes the issue
class PatchedLGBMRegressor(lightgbm.LGBMRegressor):

    @property
    def feature_names_in_(self):
        return self._feature_name

    @feature_names_in_.setter
    def feature_names_in_(self, x):
        self._feature_name = x

############################################################
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
# TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE TEMPORAIRE 
############################################################

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

    def fit(self, X, y=None):
        """
        Fit the transformer by checking for valid coordinate names and calculating the mean center.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """

        if isinstance(X, pl.DataFrame):
            self.feature_names = X.columns
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Validate the features

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pd.DataFrame or pl.DataFrame: dataframe with the right features in the right order.
        """
        
        if isinstance(X, pl.DataFrame):
            features_X = X.columns
        elif isinstance(X, pd.DataFrame):
            features_X = X.columns.tolist()

        missing_features = []
        for var in self.feature_names:
            if var not in features_X:
                missing_features += [var]
                print(f'Feature {var} is missing in the data')

        if len(missing_features) > 0:
            raise ValueError("Some features are missing in the data.")


        return X[self.feature_names]

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in one step.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        pd.DataFrame or pl.DataFrame: Transformed data.
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

    def fit(self, X, y=None):
        """
        Fit the transformer by checking for valid coordinate names and calculating the mean center.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
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

    def transform(self, X, y=None):
        """
        Apply the coordinate rotations and return the modified data.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pd.DataFrame or pl.DataFrame: Transformed data with additional rotated coordinates.
        """
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
            if isinstance(X, pl.DataFrame):
                X = X.with_columns(
                    [
                        x_temp.alias(f"{x_coord}_rotated{i}"),
                        y_temp.alias(f"{y_coord}_rotated{i}")
                    ]
                )
            if isinstance(X, pd.DataFrame):
                X[f"{x_coord}_rotated{i}"] = x_temp
                X[f"{y_coord}_rotated{i}"] = y_temp

            rotated_coordinates_names = rotated_coordinates_names + [
                f"{x_coord}_rotated{i}", f"{y_coord}_rotated{i}"
            ]

        self.rotated_coordinates_names = rotated_coordinates_names
        self.names_features_output = X.columns.tolist() if isinstance(X, pd.DataFrame) else X.columns
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in one step.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        pd.DataFrame or pl.DataFrame: Transformed data.
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

class ConvertDateToInteger(BaseEstimator, TransformerMixin):
    """
    A custom transformer to convert transaction dates to integers (days since a reference date).

    Parameters:
    transaction_date_name (str): Name of the column with transaction dates.
    reference_date (str): Reference date in YYYY-MM-DD format. Defaults to "2010-01-01".
    """
    def __init__(self, transaction_date_name: str = None, reference_date: str = "2010-01-01"):
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
        self.transaction_date_name = transaction_date_name
        self.reference_date = reference_date
        self.names_features_output = None
        return self

    def fit(self, X, y=None):
        """
        Fit the transformer by validating the transaction date column.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        transaction_date_name = self.transaction_date_name

        # Raise an error if the transaction date is not in the data
        if transaction_date_name not in X.columns:
            raise ValueError(f"Feature {transaction_date_name} is not in the data")

        # Raise an error if the transaction date is not a date
        if isinstance(X, pl.DataFrame) and not isinstance(X[transaction_date_name].dtype, pl.Date):
            raise TypeError(f"Feature {transaction_date_name} is not of type date")
        if isinstance(X, pd.DataFrame) and not pd.api.types.is_datetime64_any_dtype(X[transaction_date_name]):
            raise TypeError(f"Feature {transaction_date_name} is not of type date")

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Convert the transaction date to an integer representing the number of days since the reference date.

        Parameters:
        X (pd.DataFrame or pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pd.DataFrame or pl.DataFrame: Transformed data with integer representation of dates.
        """
        transaction_date_name = self.transaction_date_name
        reference_date = self.reference_date

        # Calculate the number of days between each date and the starting point of transaction data (January 1st, 2010)
        if isinstance(X, pl.DataFrame):
            X = X.with_columns(
                (pl.col(transaction_date_name) - pl.Series([reference_date]).str.to_date()).dt.total_days().alias(f"{transaction_date_name}")
            )
        if isinstance(X, pd.DataFrame):
            X.loc[:, f"{transaction_date_name}"] = (X[transaction_date_name] - pd.to_datetime(reference_date)).dt.days

        # Store feature names
        if isinstance(X, pl.DataFrame):
            self.names_features_output = X.columns
        if isinstance(X, pd.DataFrame):
            self.names_features_output = X.columns.tolist()
        
        # Return a Pandas dataframe because LightGBM does not accept 
        # Polars dataframes (yet)
        # BALISE: C'est ici qu'il faut passer en Pandas pour un gridsearch sans CV
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()

        return X

    def fit_transform(self, X, y=None):
        """
        Apply the transform and the fit methods to the data.
        """

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

def create_price_model_pipeline(
    model=lightgbm.LGBMRegressor(), 
    presence_coordinates = True
    ):
    """
    Create a pipeline for housing prices modelling

    Parameters:
    model (BaseEstimator, optional): Model to use for the housing prices modelling. Defaults to LGBMRegressor.
    presence_coordinates (boolean, default = True): do features contain geographical coordinates of housing units?

    Returns:
    Pipeline: The constructed pipeline.
    """
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
    A custom estimator for housing price prediction with optional target transformation.

    This model supports:
      - log-transform of the target variable,
      - normalization of price by floor area (price per square meter),
      - pipeline preprocessing of features (with or without coordinates),
      - automatic retransformation bias correction (Duan, 1983; Miller, 1984).

    Parameters
    ----------
    model : estimator, default=lightgbm.LGBMRegressor()
        The regression model to use (must follow the scikit-learn API).
    
    log_transform : bool, optional (default=None)
        If True, applies a logarithmic transformation to the target variable.

    price_sq_meter : bool, optional (default=None)
        If True, converts the target variable to price per square meter.
        Requires `floor_area_name` to be provided.

    presence_coordinates : bool, default=True
        If True, include coordinate-related preprocessing in the pipeline.

    floor_area_name : str, optional (default=None)
        Column name for floor area, required if `price_sq_meter=True`.

    Attributes
    ----------
    price_model_pipeline : sklearn.Pipeline
        The full pipeline combining preprocessing and the regression model.

    preprocessor : sklearn.TransformerMixin
        The preprocessing step of the pipeline (all steps before the model).

    model : estimator
        The final regression model from the pipeline.

    model_features : list of str
        List of feature names used for training.

    is_price_model_fitted : bool
        Indicates if the model has been fitted.

    RMSE : float
        Root mean squared error of the fitted model (if log-transform is applied).

    smearing_factor : float
        Duan’s smearing factor, used for retransformation bias correction (if log-transform is applied).

    source_RMSE : {"Train", "Val"}
        Indicates whether RMSE was computed on training or validation data.

    source_correction_terms : {"Train", "Val"}
        Indicates the data source used to compute retransformation correction factors.

    """
    def __init__(
        self,
        model=lightgbm.LGBMRegressor(),
        log_transform=None,
        price_sq_meter=None,
        presence_coordinates=True,
        floor_area_name=None
    ):

        if price_sq_meter is True and floor_area_name is None:
            raise ValueError("The model uses price per square meter, but the name of the floor area variable is missing")

        self.log_transform = log_transform
        self.price_sq_meter = price_sq_meter
        self.feature_names_in = None
        self.is_price_model_fitted = False
        self.presence_coordinates = presence_coordinates
        self.floor_area_name = floor_area_name

        print("    Initiating an unfitted price prediction pipeline.")
        self.price_model_pipeline = create_price_model_pipeline(
            model=model,
            presence_coordinates=presence_coordinates
        )
    
        self.preprocessor = self.price_model_pipeline[:-1]
        self.model        = self.price_model_pipeline[-1]

    # Pass parameters to pipeline
    def set_params(self, dico):
        self.price_model_pipeline.set_params(**dico)
        return self

    def fit(
        self,
        X, y,
        model_features = None,
        X_val = None,
        y_val = None,
        sample_weight=None,
        sample_weight_val=None,
        verbose=True,
        log_evaluation_period = 100,
        early_stopping_rounds = 25,
        **kwargs
    ):
        """
        Fit the TwoStepsModel.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            Feature matrix for training.

        y : array-like
            Target variable.

        model_features : list of str, optional
            Explicit list of feature names to use. If None, inferred from `X`.

        X_val : pd.DataFrame or pl.DataFrame, optional
            Validation feature matrix.

        y_val : array-like, optional
            Validation target variable.

        sample_weight : array-like, optional
            Sample weights for training data.

        sample_weight_val : array-like, optional
            Sample weights for validation data.

        verbose : bool, default=True
            Whether to print progress messages.

        log_evaluation_period : int, default=100
            Period for LightGBM logging callbacks.

        early_stopping_rounds : int, default=25
            Number of rounds for LightGBM early stopping.

        **kwargs : dict
            Additional arguments passed to the underlying estimator's `fit` method.

        Returns
        -------
        self : TwoStepsModel
            Fitted estimator.
        """
        # Store feature names
        if model_features is not None:
            self.model_features = model_features
        else:
            if isinstance(X, pl.DataFrame):
                self.model_features = X.columns
            if isinstance(X, pd.DataFrame):
                self.model_features = X.columns.tolist()

        print(f"    Average observed value of the dependant variable before training: {np.mean(y)}") if verbose else None

        # Transform the target according to the settings
        print("    Transforming the target") if verbose and (self.log_transform or self.price_sq_meter) else None
        y_transformed = self.transform(X, y)

        print(f"    Average observed value of the dependant variable after transformation: {np.mean(y_transformed)}") if verbose else None


        # Fit the preprocessor
        print("    Fit the preprocessor") if verbose else None
        if isinstance(X, pl.DataFrame):
            self.preprocessor.fit(X.select(model_features), y)
        elif isinstance(X, pd.DataFrame):
            self.preprocessor.fit(X[model_features], y)

        # Transform the data with the preprocessor
        print("    Transform training data with the preprocessor") if verbose else None
        if isinstance(X, pl.DataFrame):
            X_transformed = self.preprocessor.transform(X.select(model_features))
        elif isinstance(X, pd.DataFrame):
            X_transformed = self.preprocessor.transform(X[model_features])

        # Prepare validation data is any
        if X_val is not None and y_val is not None:
            print("    Transform validation data") if verbose else None
            if isinstance(X, pl.DataFrame):
                X_val_transformed = self.preprocessor.transform(X_val.select(model_features))
            elif isinstance(X, pd.DataFrame):
                X_val_transformed = self.preprocessor.transform(X_val[model_features])
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
        if str(self.price_model_pipeline[-1].__class__) in [
            "<class 'lightgbm.sklearn.LGBMRegressor'>",
            "<class 'functions.modelling_functions.PatchedLGBMRegressor'>"
        ]:
            callbacks=[
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

        elif str(self.price_model_pipeline[-1].__class__) == "<class 'sklearnex.ensemble._forest.RandomForestRegressor'>":
            print("    Training the model") if verbose else None
            self.price_model_pipeline[-1].fit(
                X_transformed,
                y_transformed,
                sample_weight=sample_weight
            )            

        end_time = time.monotonic()

        print(f"    Training time of the price prediction model: {end_time - start_time} seconds") if verbose else None

        if self.log_transform:
            # Compute the model's RMSE and correction term (useful for the retransformation correction)
            print("    Compute the model's correction terms") if verbose else None
            if X_val is not None and y_val is not None:
                y_pred = self.price_model_pipeline.predict(X_val)
                self.RMSE = math.sqrt(metrics.mean_squared_error(y_val_transformed, y_pred))
                self.smearing_factor = np.mean(np.exp(y_val_transformed - y_pred))
                self.source_RMSE = "Val"
                self.source_correction_terms = "Val"
            else:
                y_pred = self.price_model_pipeline.predict(X)
                self.RMSE = math.sqrt(metrics.mean_squared_error(y_transformed, y_pred))
                self.smearing_factor = np.mean(np.exp(y_transformed - y_pred))
                self.source_RMSE = "Train"
                self.source_correction_terms = "Train"

            print("    RMSE = ", self.RMSE)
            print("    Smearing factor = ", self.smearing_factor)

        self.is_price_model_fitted = True

        return self

    def transform(self, X, y):
        """
        Apply target transformation according to the model specification.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            Feature matrix, used if `price_sq_meter=True`.

        y : array-like
            Target values to transform.

        Returns
        -------
        y_transformed : np.ndarray
            Transformed target values.
        """
        y_transform = np.copy(y)

        # Compute the price per square meter
        if self.price_sq_meter:
            y_transform = y_transform / X[self.floor_area_name].to_numpy()
        # Take the logarithm
        if self.log_transform:
            y_transform = np.log(y_transform)
        return y_transform

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

        """
        Predict with the fitted model.

        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            Feature matrix.

        iteration_range : tuple, optional
            Iteration range for boosting models (LightGBM only).

        add_retransformation_correction : bool, default=True
            Whether to apply retransformation bias correction if `log_transform=True`.

        retransformation_method : {"Duan", "Miller"}, default="Duan"
            Method for retransformation correction.

        verbose : bool, default=True
            Whether to print progress messages.

        **kwargs : dict
            Additional arguments passed to the underlying estimator's `predict` method.

        Returns
        -------
        y_pred : np.ndarray
            Predicted values in the original target scale.
        """

        if add_retransformation_correction is True and retransformation_method not in ["Duan", "Miller"]:
            raise ValueError("The retransformation_method argument must be either features 'Duan' or 'Miller'.")

        # Predict the local average
        print("    Predicting the target") if verbose else None
        y_pred = self.price_model_pipeline.predict(X)

        # Invert the target transformation
        print("    Invert the target transformation") if verbose and (self.price_sq_meter or self.log_transform) else None
        y_pred = self.inverse_transform(X, y_pred)

        if self.log_transform:
            if add_retransformation_correction:
                print("    The models includes a correction of the retransformation bias.") if verbose else None
                if retransformation_method == "Duan":
                    print("    The retransformation bias is corrected using Duan's 1983 smearing factor.") if verbose else None
                    # Use the Duan's 1983 smearing factor correction
                    global_correction = self.smearing_factor
                if retransformation_method == "Miller":
                    print("    The retransformation bias is corrected using Miller's 1984 correction factor.") if verbose else None
                    # Use the Miller's 1984 retransformation correction
                    global_correction = np.exp((self.RMSE ** 2) / 2)
                print("    Average correction = ", round(100 * (global_correction - 1), 2), '%')
                y_pred = y_pred * global_correction
            else:
                print("    There is no correction for the retransformation bias.") if verbose else None
        else:
            print("    The model has no log-transformation.") if verbose else None

        return y_pred

def predict_market_value(
    X: pl.DataFrame,
    model,
    date_market_value = None,
    add_retransformation_correction = False,
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
        raise ValueError(f"Data should not contain the column {transaction_date_name} if date_market_value is not None.")

    if date_market_value is not None:
        print(f'    Predicting market values at date {date_market_value}.')
        X = (
            X
            .with_columns(
                pl.lit(date_market_value).str.to_date(format = '%Y-%m-%d').alias(transaction_date_name),
                pl.lit(date_market_value[0:4]).str.to_integer().alias("anneemut"),
                pl.lit(date_market_value[5:7]).str.to_integer().alias("moismut")
            )
        )
    elif transaction_date_name in data.columns:
        print(f'    Predicting market values using transaction date from the data.')
    else:
        raise ValueError("The date for market value prediction is missing.")
    
    # Predict market values
    market_values = model.predict(X, **kwargs)

    return market_values
