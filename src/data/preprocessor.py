import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import pickle

from src.data.geo_data_handler import GeoDataHandler
from src.utils.tile_manager import TileManager
from src.models.cnf_utils import CNFUtils
from src.models.normalizing_flow import ConditionalNormalizingFlow


class WildfireDataProcessor:
    def __init__(self, pyrome_shapefile_path='wildfire-ignition-generator/data/pyrome_shp/Pyromes_CONUS_20200206.shp', pyrome_classes_path='wildfire-ignition-generator/data/vars_50k/pyrome_np.pkl'):
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()
        self.pyrome_stats = {}
        self.fit_performed = False
        
        print('Loading pyrome files ...')
        self.load_pyrome_shapefile(pyrome_shapefile_path)
        self.load_valid_pyromes(pyrome_classes_path)
        self.geo_handler = GeoDataHandler()
        self.pyrome_classifier = self.create_pyrome_classifier(pyrome_shapefile_path)

    def create_pyrome_classifier(self, pyrome_shapefile_path):
        """
        Create a function that classifies lat/long coordinates to pyromes.

        Args:
            pyrome_shapefile_path (str): Path to the pyrome shapefile.

        Returns:
            callable: A function that takes lat/long and returns the pyrome.
        """
        pyromes_gdf = gpd.read_file(pyrome_shapefile_path)
        pyromes_gdf = pyromes_gdf.to_crs(epsg=4326)  # Ensure WGS84 projection

        def classify_coords(lat, lon):
            point = Point(lon, lat)
            for idx, row in pyromes_gdf.iterrows():
                if row['geometry'].contains(point):
                    return row['PYROME']
            return None

        return classify_coords
        
    def process_daily_data(self, geo_handler, date):
        """
        Process data for a specific day.

        Args:
            geo_handler (GeoDataHandler): An instance of GeoDataHandler.
            date (datetime): The date for which to process data.

        Returns:
            torch.Tensor: Processed data ready for model inference.
        """
        # Get the daily DataFrame
        df = geo_handler.get_daily_dataframe(date, self.pyrome_classifier)

        # Preprocess the data
        processed_df = self._preprocess_data(df, fit=False)

        # Prepare for training (convert to tensor)
        X, _ = self.prepare_for_training(processed_df, is_training=False)

        return X

    def process_point(self, lat, lon, time, erc, bi):
        pyrome = self.classify_coords(lat, lon)
        erc_percentile = self.convert_to_percentile(erc, 'ERC', pyrome)
        bi_percentile = self.convert_to_percentile(bi, 'BI', pyrome)
        # ... (process other features)
        return self.prepare_for_model(features)

    def convert_to_percentile(self, value, feature, pyrome):
        stats = self.pyrome_stats[feature][pyrome]
        return np.interp(value, [stats['min'], stats['max']], [0, 100])

    def prepare_for_model(self, features):
        # Scale features and convert to tensor
        return torch.tensor(self.scaler.transform(features), dtype=torch.float32)

    def load_valid_pyromes(self, pyrome_classes_path):
        def load_np_array(pickle_file_path, nodata_value=-9999):
            with open(pickle_file_path, 'rb') as f:
                np_array = pickle.load(f)
            np_array = np.where(np_array == nodata_value, np.nan, np_array)
            return np_array

        self.pyromes_unique = load_np_array(pyrome_classes_path)
        self.pyromes_unique = np.unique(self.pyromes_unique[~np.isnan(self.pyromes_unique)])
        print(f'Unique Pyrome: {self.pyromes_unique}')

    def load_pyrome_shapefile(self, shapefile_path):
        self.pyromes_gdf = gpd.read_file(shapefile_path)
        self.pyromes_gdf = self.pyromes_gdf.to_crs(epsg=4326)  # Ensure WGS84 projection

    def classify_coords(self, latitude, longitude):
        point = Point(longitude, latitude)
        for idx, row in self.pyromes_gdf.iterrows():
            if row['geometry'].contains(point):
                return row['PYROME']
        return None

    def utm_to_latlon(self, easting, northing, zone, northern=True):
        utm_crs = pyproj.CRS.from_string(f"+proj=utm +zone={zone} +{'north' if northern else 'south'} +ellps=WGS84")
        wgs84_crs = pyproj.CRS('EPSG:4326')
        transform = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
        lon, lat = transform.transform(easting, northing)
        return lat, lon

    def classify_utm_coords(self, easting, northing, zone, northern=True):
        lat, lon = self.utm_to_latlon(easting, northing, zone, northern)
        return self.classify_coords(lat, lon)

    def fit(self, df):
        self.fit_performed = True
        return self._preprocess_data(df, fit=True)

    def transform(self, df):
        if not self.fit_performed:
            raise ValueError("Fit must be performed before transform")
        return self._preprocess_data(df, fit=False)

    def fit_transform(self, df):
        return self.fit(df)

    def _preprocess_data(self, df, fit=True):
        def print_nan_info(df, step_name):
            nan_count = df.isna().sum().sum()
            print(f"\n{step_name}:")
            print(f"Total NaN count: {nan_count}")
            
            if nan_count > 0:
                nan_percentages = (df.isna().sum() / len(df)) * 100
                print("\nPercentage of NaN values per column:")
                for col, percentage in nan_percentages[nan_percentages > 0].items():
                    print(f"{col}: {percentage:.2f}%")
                
                print("\nFirst 10 rows with NaN values:")
                print(df[df.isna().any(axis=1)].head(10))

        print_nan_info(df, "Initial DataFrame")
        # Add original_index column
        df['original_index'] = df.index

        if fit is False and 'Acres' not in df.columns:
            df['Acres'] = 0 

            
        # Convert date columns to datetime format
        df['Ignition date'] = pd.to_datetime(df['Ignition date'])
        #df['Containment date'] = pd.to_datetime(df['Containment date'])
        #print_nan_info(df, "After date conversion")

        # Calculate fire duration and select relevant columns
        #df['Duration'] = (df['Containment date'] - df['Ignition date']).dt.days
        df = df[['Latitude', 'Longitude', 'ERC', 'WUI proximity', 'BI', 'Pyrome', 'Acres', 'Ignition date', 'original_index']]
        print_nan_info(df, "After selecting relevant columns")

        # Filter out the dataset to include only the Western United States
        df = df[df['Pyrome'].isin(np.unique(self.pyromes_unique))]
        print_nan_info(df, "After filtering for Western US")

        # Handle missing values
        for col in df.columns:
            if df[col].dtype != 'object' and col != 'Ignition date':
                df[col] = df[col].fillna(df[col].median())
        df = df.dropna()
        print_nan_info(df, "After handling missing values")

        # Convert ERC, WUI, and BI to percentiles within each pyrome
        for col in ['ERC', 'WUI proximity', 'BI']:
            if fit:
                self.pyrome_stats[col] = df.groupby('Pyrome')[col].describe()
            df[f'{col}_percentile'] = df.groupby('Pyrome')[col].rank(pct=True) * 100
        print_nan_info(df, "After percentile conversion")

        # One-hot encode the pyrome column
        if fit:
            pyrome_encoded = self.encoder.fit_transform(df[['Pyrome']]).toarray()
        else:
            pyrome_encoded = self.encoder.transform(df[['Pyrome']]).toarray()
       
        pyrome_df = pd.DataFrame(pyrome_encoded, columns=self.encoder.get_feature_names_out(['Pyrome']))
        print(f"Number of rows before one-hot encoding: {len(df)}")

        # One-hot encode the pyrome column using pandas
        pyrome_df = pd.get_dummies(df['Pyrome'], prefix='Pyrome').astype(int)

        print(pyrome_df.head())
        # Compute cyclic date variables
        df['ignition_sin'] = df['Ignition date'].apply(self._date_sin)
        df['ignition_cos'] = df['Ignition date'].apply(self._date_cos)
        print_nan_info(df, "After computing cyclic date variables")

        # Combine all features into a single dataframe
        features_to_combine = ['Latitude', 'Longitude', 'ERC_percentile', 'WUI proximity_percentile', 'BI_percentile', 'Acres', 'ignition_sin', 'ignition_cos', 'Ignition date', 'original_index']
        model_df = pd.concat([df[features_to_combine], pyrome_df], axis=1)   
        print_nan_info(model_df, "After combining features")

        # Standardize continuous variables
        continuous_vars = model_df[['Latitude', 'Longitude', 'ERC_percentile', 'WUI proximity_percentile', 'BI_percentile', 'ignition_sin', 'ignition_cos']]
        if fit:
            scaled_vars = self.scaler.fit_transform(continuous_vars)
        else:
            scaled_vars = self.scaler.transform(continuous_vars)

        scaled_df = pd.DataFrame(scaled_vars, columns=['Latitude', 'Longitude', 'ERC_percentile', 'WUI proximity_percentile', 'BI_percentile', 'ignition_sin', 'ignition_cos'])
        print('scaled_df')
        print(scaled_df.head)
        print_nan_info(scaled_df, "After scaling")
        acres_df=model_df[['Acres', 'Ignition date', 'original_index']]
        print('acres_df')
        print(acres_df.head)

        print(f"Number of rows with NaNs in scaled_df: {scaled_df.isna().sum().sum()}, pyrome_df: {pyrome_df.isna().sum().sum()}, acres_df: {acres_df.isna().sum().sum()}")
        print(f"Size of scaled_df: {scaled_df.shape[0]}, pyrome_df: {pyrome_df.shape[0]}, acres_df: {acres_df.shape[0]}")

        scaled_df.reset_index(drop=True, inplace=True)
        pyrome_df.reset_index(drop=True, inplace=True)
        acres_df.reset_index(drop=True, inplace=True)

        # Combine scaled continuous variables and categorical variables
        model_df = pd.concat([scaled_df, pyrome_df, acres_df], axis=1)
        print('model_df ')
        print(model_df.shape)
        print(model_df.head)

        print_nan_info(model_df, "Final DataFrame before filtering")

        # Filter out records with Acres <= 100
        if fit:
            model_df = model_df[model_df['Acres'] > 100]

        print_nan_info(model_df, "Final DataFrame after filtering")

        return model_df

    @staticmethod
    def _date_sin(date):
        return np.sin(2 * np.pi * date.timetuple().tm_yday / 366.)

    @staticmethod
    def _date_cos(date):
        return np.cos(2 * np.pi * date.timetuple().tm_yday / 366.)

    def split_data(self, df, years=3):
        # Convert 'Ignition date' to datetime if not already
        print("Converting 'Ignition date' to datetime format if necessary.")
        df['Ignition date'] = pd.to_datetime(df['Ignition date'])
        print("Converted 'Ignition date'.")
        print(f"First few entries:\n{df['Ignition date'].head()}")

        # Determine the cutoff date
        cutoff_date = df['Ignition date'].max() - pd.DateOffset(years=years)
        print(f"Determined cutoff date: {cutoff_date}")

        # Split the data
        print("Splitting the data into training and validation sets.")
        train_df = df[df['Ignition date'] <= cutoff_date]
        val_df = df[df['Ignition date'] > cutoff_date]

        # Print beginning and end dates for training and validation sets
        print(f"Training set range: {train_df['Ignition date'].min()} to {train_df['Ignition date'].max()}")
        print(f"Validation set range: {val_df['Ignition date'].min()} to {val_df['Ignition date'].max()}")

        print(f"Training set size: {train_df.shape[0]} records")
        print(f"Validation set size: {val_df.shape[0]} records")

        print('-'*30)
        print('Train Dataframe:')
        print(train_df.head(20))
        print('-'*30)
        print('Val Dataframe:')
        print(val_df.head(20))
        print('-'*30)

        # Print rows with NaN values if any
        if train_df.isnull().values.any():
            print("Rows with NaN values in the training set:")
            print(train_df[train_df.isnull().any(axis=1)])
        
        if val_df.isnull().values.any():
            print("Rows with NaN values in the validation set:")
            print(val_df[val_df.isnull().any(axis=1)])

        # Check for NaN values
        print(f"NaN values in training set: {train_df.isnull().sum().sum()}")
        print(f"NaN values in validation set: {val_df.isnull().sum().sum()}")

        # Ensure there are no NaN values
        assert not train_df.isnull().values.any(), "Training set contains NaN values."
        assert not val_df.isnull().values.any(), "Validation set contains NaN values."

        return train_df, val_df

    def prepare_for_training(self, df, is_training=True):
        # Drop the target and date columns
        X = df.drop(columns=['Acres', 'Ignition date'])
        y = df['Acres']
        
        # Print shapes after dropping columns
        print(f"Shape of features (X) after dropping 'Acres' and 'Ignition date': {X.shape}")
        print(f"Shape of target (y) after selection: {y.shape}")

        # Log transform the target variable
        y = np.log1p(y)
        print("Target variable log-transformed.")

        # Convert to numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # Print the heads of the arrays
        print("First few entries of X:")
        print(X[:5])
        print("First few entries of y:")
        print(y[:5])

        # Ensure there are no NaN values
        assert not np.isnan(X).any(), "Features contain NaN values."
        assert not np.isnan(y).any(), "Target contains NaN values."

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Print shapes after conversion to PyTorch tensors
        print(f"Shape of features (X) after conversion to PyTorch tensor: {X.shape}")
        print(f"Shape of target (y) after conversion to PyTorch tensor: {y.shape}")

        return X, y

    def save(self, path='wildfire-ignition-generator/data/processed/processor'):
        if not self.fit_performed:
            raise ValueError("Fit must be performed before saving")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.encoder, os.path.join(path, 'encoder.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        joblib.dump(self.pyrome_stats, os.path.join(path, 'pyrome_stats.joblib'))
        print('saved encoders!')

    def load(self, path='wildfire-ignition-generator/data/processed/processor'):
        self.encoder = joblib.load(os.path.join(path, 'encoder.joblib'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
        self.pyrome_stats = joblib.load(os.path.join(path, 'pyrome_stats.joblib'))
        self.fit_performed = True
        print('loaded encoders!')

    def process_new_data(self, lat, lon, erc, wui, bi, date):
        pyrome = self.classify_coords(lat, lon)
        if pyrome is None or pyrome not in self.valid_pyromes:
            return None

        # Convert to percentiles
        erc_percentile = np.interp(erc, self.pyrome_stats['ERC'].loc[pyrome, ['min', 'max']], [0, 100])
        wui_percentile = np.interp(wui, self.pyrome_stats['WUI proximity'].loc[pyrome, ['min', 'max']], [0, 100])
        bi_percentile = np.interp(bi, self.pyrome_stats['BI'].loc[pyrome, ['min', 'max']], [0, 100])

        # Prepare date features
        date = pd.to_datetime(date)
        ignition_sin = self._date_sin(date)
        ignition_cos = self._date_cos(date)

        # Prepare feature vector
        features = np.array([[lat, lon, erc_percentile, wui_percentile, bi_percentile, ignition_sin, ignition_cos]])
        
        # One-hot encode pyrome
        pyrome_encoded = self.encoder.transform([[pyrome]]).toarray()

        # Combine and scale features
        features_combined = np.hstack((features, pyrome_encoded))
        features_scaled = self.scaler.transform(features_combined)

        return torch.tensor(features_scaled, dtype=torch.float32)

# Usage example
if __name__ == "__main__":
    # Load your data here and use the processor
    pass
