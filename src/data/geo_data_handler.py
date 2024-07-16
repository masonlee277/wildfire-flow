"""
GeoDataHandler: A comprehensive class for handling geospatial data operations.

This class provides methods for loading, processing, and querying various types of
geospatial data, including GeoTIFF rasters, NetCDF files, and shapefiles. It also
includes functionality for reprojection, raster querying, and file format conversions.

Key features:
- Load and save GeoTIFF rasters
- Load and process NetCDF files
- Convert NetCDF to GeoTIFF
- Reproject raster data
- Query raster values at specific coordinates
- Load and process shapefiles
- Calculate zonal statistics

Dependencies:
- rasterio
- xarray
- geopandas
- numpy
- pyproj
- rasterstats

Usage:
    handler = GeoDataHandler()
    raster_data, metadata = handler.load_geotiff('path/to/file.tif')
    value = handler.query_raster_value(raster_data, metadata, lon, lat)
    handler.netcdf_to_geotiff('input.nc', 'output.tif', 'variable_name')

Note: Ensure all required libraries are installed and properly configured.
"""

import rasterio
import xarray as xr
import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pyproj import CRS, Transformer
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import os
import pandas as pd 
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
from shapely.geometry import Point
from rtree import index
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask.dataframe as dd
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from multiprocessing import Pool, cpu_count
from pyproj import CRS, Transformer
import math

class GeoDataHandler:
    def __init__(self, data_dir='data/wildfire_indicators/raw_netcdfs', pyrome_shapefile_path='data/pyrome_shp/Pyromes_CONUS_20200206.shp'):
        self.data_dir = data_dir
        self.netcdf_data = {}
        self.wui_data = None
        self.wui_meta = None
        self.variable_names = {}  # Store the correct variable names
        self.pyrome_classifier = None
        self.pyrome_index = None
        self.pyrome_gdf = None
        #self.pyrome_classifier = self.create_pyrome_classifier(pyrome_shapefile_path)
        self.load_pyrome_shapefile(pyrome_shapefile_path)



    def load_netcdf_data(self, variable, year):
        """Load NetCDF data for a specific variable and year."""
        file_path = os.path.join(self.data_dir, f"{variable}_{year}.nc")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NetCDF file not found: {file_path}")
        print(f"Loading NetCDF file: {file_path}")

        dataset = xr.open_dataset(file_path)
        self.netcdf_data[f"{variable}_{year}"] = dataset
        
        # Detect the correct variable name
        if variable == 'erc':
            possible_names = ['energy_release_component-g', 'energy_release_component_g']
        elif variable == 'bi':
            possible_names = ['burning_index-g', 'burning_index_g']
        else:
            raise ValueError(f"Unknown variable: {variable}")

        for name in possible_names:
            if name in dataset.variables:
                self.variable_names[variable] = name
                break
        else:
            raise KeyError(f"No matching variable found for {variable}. Available variables: {list(dataset.variables.keys())}")

        print(f"Detected variable name for {variable}: {self.variable_names[variable]}")

    def get_variable_name(self, variable):
        """Get the correct variable name for a given variable type."""
        if variable not in self.variable_names:
            # If the variable name hasn't been detected yet, try to detect it
            self.detect_variable_name(variable)
        return self.variable_names[variable]

    def detect_variable_name(self, variable):
        """Detect the correct variable name from the NetCDF data."""
        key = f"{variable}_2023"  # Assuming we're always working with 2023 data
        if key not in self.netcdf_data:
            raise KeyError(f"NetCDF data for {variable} not loaded. Call load_netcdf_data first.")

        if variable == 'erc':
            possible_names = ['energy_release_component-g', 'energy_release_component_g', 'erc']
        elif variable == 'bi':
            possible_names = ['burning_index-g', 'burning_index_g', 'bi']
        else:
            raise ValueError(f"Unknown variable: {variable}")

        for name in possible_names:
            if name in self.netcdf_data[key].variables:
                self.variable_names[variable] = name
                print(f"Detected variable name for {variable}: {name}")
                return

        raise KeyError(f"No matching variable found for {variable}. Available variables: {list(self.netcdf_data[key].variables.keys())}")

    def create_pyrome_classifier(self, pyrome_shapefile_path):
        """
        Create an efficient pyrome classifier using R-tree spatial indexing.
        """
        print("Loading pyrome shapefile...")
        self.pyrome_gdf = gpd.read_file(pyrome_shapefile_path)
        self.pyrome_gdf = self.pyrome_gdf.to_crs(epsg=4326)  # Ensure WGS84 projection
        
        print("Creating R-tree spatial index...")
        self.pyrome_index = index.Index()
        for idx, geometry in enumerate(self.pyrome_gdf.geometry):
            self.pyrome_index.insert(idx, geometry.bounds)

        print("Pyrome classifier ready.")

    def classify_coords(self, lat, lon):
        point = Point(lon, lat)
        potential_matches_idx = list(self.pyrome_index.intersection(point.bounds))
        for idx in potential_matches_idx:
            if self.pyrome_gdf.iloc[idx].geometry.contains(point):
                return self.pyrome_gdf.iloc[idx]['PYROME']
        return None

    @staticmethod
    def _classify_chunk(chunk, classifier):
        lats, lons = chunk
        return [classifier(lat, lon) for lat, lon in zip(lats, lons)]


    def get_daily_dataframe(self, date, n_processes=None):
        """Create a DataFrame with ERC, BI, WUI, lat/long, and pyrome for each pixel for a specific date."""
        year = date.year

        erc_key = f"erc_{year}"
        bi_key = f"bi_{year}"
        print(f"Using {n_processes} processes for parallel execution.")
        
        if erc_key not in self.netcdf_data:
            self.load_netcdf_data("erc", year)
        if bi_key not in self.netcdf_data:
            self.load_netcdf_data("bi", year)
        if self.wui_data is None:
            self.load_wui_data("placeholder_wui.tif")
        print('finished')
        
        day_of_year = date.timetuple().tm_yday - 1
        print(f"Processing data for day {day_of_year} of year {year}.")

        erc_var_name = self.get_variable_name('erc')
        bi_var_name = self.get_variable_name('bi')

        print(erc_var_name)

        erc_data = self.netcdf_data[erc_key][erc_var_name][day_of_year].values
        bi_data = self.netcdf_data[bi_key][bi_var_name][day_of_year].values

        lats = self.netcdf_data[erc_key]['lat'].values
        lons = self.netcdf_data[erc_key]['lon'].values

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        erc_flat = erc_data.flatten()
        bi_flat = bi_data.flatten()
        wui_flat = self.wui_data.flatten()
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()

        num_pixels = erc_flat.size
        max_x = erc_data.shape[1] - 1
        max_y = erc_data.shape[0] - 1

        print(f'Number of pixels: {num_pixels}')
        print(f'Image size (X, Y): ({max_x}, {max_y})')
        print(f'Number of Rows to Process: {erc_flat.shape}')
        df = pd.DataFrame({
            'Latitude': lat_flat,
            'Longitude': lon_flat,
            'ERC': erc_flat,
            'BI': bi_flat,
            'WUI proximity': wui_flat,
            'X': np.repeat(np.arange(erc_data.shape[1]), erc_data.shape[0]),
            'Y': np.tile(np.arange(erc_data.shape[0]), erc_data.shape[1])
        })
        print("Initial DataFrame created, starting parallel classification.")
        print('init df created')

        df = df.dropna()
        print(df.head())
        print(f'DataFrame shape after dropping NaN values: {df.shape}')

        print('Classifying pyromes...')

        df = df.dropna()
        print(f'DataFrame shape after dropping NaN values: {df.shape}')

        # Use the efficient pyrome classification method
        df = self.classify_pyromes_efficient(df)
        nan_count = df['Pyrome'].isna().sum()
        print(f"Number of NaN values in 'Pyrome' column: {nan_count}")
        print(f"DataFrame shape after pyrome classification: {df.shape}")

        df = df.dropna()
        df['Ignition date'] = date


        print(f"Final DataFrame shape after pyrome classification: {df.shape}")
        print(df.head())
        return df

    def load_pyrome_shapefile(self, pyrome_shapefile_path):
        """Load the pyrome shapefile and create a spatial index."""
        print("Loading pyrome shapefile...")
        self.pyrome_gdf = gpd.read_file(pyrome_shapefile_path)
        self.pyrome_gdf = self.pyrome_gdf.to_crs(epsg=4326)  # Ensure WGS84 projection
        
        # Create a spatial index
        self.pyrome_gdf = self.pyrome_gdf.sort_values(['PYROME'])
        self.pyrome_gdf['geometry'] = self.pyrome_gdf.geometry.buffer(0)  # Fix any invalid geometries
        self.pyrome_sindex = self.pyrome_gdf.sindex
        
        print("Pyrome shapefile loaded and indexed.")
        
        # Visualize the pyromes
        self.visualize_pyromes()
    
    def classify_pyromes_efficient(self, df):
        """Efficiently classify pyromes for a DataFrame of points."""
        print("Classifying pyromes efficiently...")
        
        # Create a GeoDataFrame from the input DataFrame
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
        gdf = gdf.set_crs(epsg=4326)
        
        # Perform a spatial join
        joined = gpd.sjoin(gdf, self.pyrome_gdf, how='left', predicate='within')
        
        # Transfer the PYROME values back to the original DataFrame
        df['Pyrome'] = joined['PYROME']
        
        
        print("Pyrome classification complete.")
        return df

    def visualize_pyromes(self):
        """Visualize the pyromes with their numbers in the center."""
        fig, ax = plt.subplots(figsize=(15, 10))
        self.pyrome_gdf.plot(ax=ax, edgecolor='black', alpha=0.5)
        
        for idx, row in self.pyrome_gdf.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(text=row['PYROME'], xy=(centroid.x, centroid.y), 
                        xytext=(3, 3), textcoords="offset points", 
                        fontsize=8, color='red')
        
        plt.title('Pyromes')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def classify_dataframe(self, df_chunk):
        """Classify a chunk of the DataFrame."""
        df_chunk['Pyrome'] = df_chunk.apply(lambda row: self.classify_coords(row['Latitude'], row['Longitude']), axis=1)
        print(df_chunk.head())
        return df_chunk
    
    def get_netcdf_value(self, variable, year, date, lon, lat):
        """Get a value from NetCDF data for a specific variable, date, and location."""
        key = f"{variable}_{year}"
        if key not in self.netcdf_data:
            self.load_netcdf_data(variable, year)
        
        data = self.netcdf_data[key]
        day_of_year = date.timetuple().tm_yday - 1  # xarray uses 0-based indexing
        return data[variable].sel(day=day_of_year, lon=lon, lat=lat, method='nearest').item()

    def get_wui_value(self, lon, lat):
        """Get a WUI value for a specific location."""
        if self.wui_data is None:
            raise ValueError("WUI data not loaded. Call load_wui_data first.")
        
        row, col = self._get_pixel_indices(lon, lat, self.wui_meta['transform'])
        return self.wui_data[row, col]

    def _get_pixel_indices(self, lon, lat, transform):
        """Convert lon/lat to pixel indices."""
        col, row = ~transform * (lon, lat)
        return int(row), int(col)

    def get_key_var(self, variable):
        if variable=='erc': key_var = 'energy_release_component-g'
        elif variable=='bi': key_var = 'burning_index-g'
        else: key_var = variable
        return key_var

    def create_placeholder_wui(self, variable='erc', year=2023):
        """
        Create a placeholder WUI file using the first band of the ERC NetCDF.
        
        Args:
            variable (str): Variable to use for creating the placeholder (default is 'erc')
            year (int): Year of the data to use (default is 2023)
        """
        key = f"{variable}_{year}"
        if key not in self.netcdf_data:
            self.load_netcdf_data(variable, year)
        
        data = self.netcdf_data[key]
        var_name = 'energy_release_component-g' if variable == 'erc' else 'burning_index-g'
        
        # Use the first band (first day) of the data
        wui_data = data[var_name][0].values
        
        # Get spatial information
        lons = data.lon.values
        lats = data.lat.values
        
        # Create the GeoTIFF
        transform = from_origin(lons.min(), lats.max(), 
                                lons[1] - lons[0], lats[0] - lats[1])
        
        wui_path = os.path.join(self.data_dir, 'placeholder_wui.tif')
        
        with rasterio.open(
            wui_path,
            'w',
            driver='GTiff',
            height=wui_data.shape[0],
            width=wui_data.shape[1],
            count=1,
            dtype=wui_data.dtype,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=transform,
        ) as dst:
            dst.write(wui_data, 1)
        
        print(f"Placeholder WUI file created at: {wui_path}")
        
        # Load the created WUI file
        self.load_wui_data('placeholder_wui.tif')

    def load_wui_data(self, file_name):
        """Load WUI data from a GeoTIFF file."""
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"WUI file not found. Creating placeholder WUI.")
            self.create_placeholder_wui()
            return

        with rasterio.open(file_path) as src:
            self.wui_data = src.read(1)
            self.wui_meta = src.meta
        
    def visualize_netcdf_data(self, variable, year, date):
        """Visualize NetCDF data for a specific variable, year, and date."""
        key_var  = self.get_key_var(variable)
        key = f"{variable}_{year}"
        if key not in self.netcdf_data:
            self.load_netcdf_data(variable, year)
        
        data = self.netcdf_data[key]
        day_of_year = date.timetuple().tm_yday - 1
        
        plt.figure(figsize=(14, 6))
        data[key_var][day_of_year].plot()
        plt.title(f"{variable.upper()} - {date}")
        plt.show()

    def visualize_wui_data(self):
        """Visualize WUI data."""
        if self.wui_data is None:
            raise ValueError("WUI data not loaded. Call load_wui_data first.")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(self.wui_data, cmap='viridis')
        plt.colorbar(label='WUI')
        plt.title("Wildland-Urban Interface (WUI)")
        plt.show()

    def preprocess_data(self, variable, year, output_dir):
        """Preprocess NetCDF data and save as GeoTIFF files."""
        key = f"{variable}_{year}"
        if key not in self.netcdf_data:
            self.load_netcdf_data(variable, year)
        
        data = self.netcdf_data[key]
        os.makedirs(output_dir, exist_ok=True)
        
        for day in range(data[variable].shape[0]):
            band = data[variable][day]
            output_path = os.path.join(output_dir, f"{variable}_{year}_{day+1:03d}.tif")
            
            transform = from_origin(
                data.lon.min(), data.lat.max(),
                (data.lon.max() - data.lon.min()) / data.lon.size,
                (data.lat.max() - data.lat.min()) / data.lat.size
            )
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=band.shape[0],
                width=band.shape[1],
                count=1,
                dtype=band.dtype,
                crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                transform=transform,
            ) as dst:
                dst.write(band.values, 1)

    def close(self):
        """Close all open datasets."""
        for dataset in self.netcdf_data.values():
            dataset.close()
        self.netcdf_data.clear()

    def load_geotiff(self, file_path):
        """
        Load a GeoTIFF file.

        Args:
            file_path (str): Path to the GeoTIFF file.

        Returns:
            tuple: (numpy.array, dict) Raster data and metadata.
        """
        with rasterio.open(file_path) as src:
            return src.read(1), src.meta

    def save_geotiff(self, file_path, data, metadata):
        """
        Save data as a GeoTIFF file.

        Args:
            file_path (str): Path to save the GeoTIFF file.
            data (numpy.array): Raster data to save.
            metadata (dict): Metadata for the raster.
        """
        with rasterio.open(file_path, 'w', **metadata) as dst:
            dst.write(data, 1)

    def load_netcdf(self, file_path):
        """
        Load a NetCDF file.

        Args:
            file_path (str): Path to the NetCDF file.

        Returns:
            xarray.Dataset: Loaded NetCDF dataset.
        """
        return xr.open_dataset(file_path)

    def netcdf_to_geotiff(self, netcdf_path, geotiff_path, variable, time_index=0):
        """
        Convert a NetCDF file to GeoTIFF format.

        Args:
            netcdf_path (str): Path to the input NetCDF file.
            geotiff_path (str): Path to save the output GeoTIFF file.
            variable (str): Name of the variable to extract from NetCDF.
            time_index (int, optional): Time index to use for 3D data. Defaults to 0.
        """
        ds = self.load_netcdf(netcdf_path)
        data = ds[variable].values

        # Handle 3D data (assuming first dimension is time)
        if data.ndim == 3:
            data = data[time_index, :, :]

        # Get spatial information
        lons, lats = np.meshgrid(ds['longitude'], ds['latitude'])
        
        # Create the GeoTIFF
        transform = rasterio.transform.from_bounds(
            lons.min(), lats.min(), lons.max(), lats.max(), data.shape[1], data.shape[0]
        )
        
        with rasterio.open(
            geotiff_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    def reproject_raster(self, data, src_meta, dst_crs):
        """
        Reproject a raster to a new coordinate reference system.

        Args:
            data (numpy.array): Input raster data.
            src_meta (dict): Metadata of the source raster.
            dst_crs (str): Destination CRS in EPSG code (e.g., 'EPSG:4326').

        Returns:
            tuple: (numpy.array, dict) Reprojected data and updated metadata.
        """
        src_crs = src_meta['crs']
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src_meta['width'], src_meta['height'], 
            *rasterio.transform.array_bounds(src_meta['height'], src_meta['width'], src_meta['transform'])
        )
        
        dst_meta = src_meta.copy()
        dst_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        dst_data = np.zeros((height, width), dtype=src_meta['dtype'])
        
        reproject(
            source=data,
            destination=dst_data,
            src_transform=src_meta['transform'],
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        
        return dst_data, dst_meta

    def query_raster_value(self, data, metadata, lon, lat):
        """
        Query raster value at a specific longitude and latitude.

        Args:
            data (numpy.array): Raster data.
            metadata (dict): Raster metadata.
            lon (float): Longitude of the point.
            lat (float): Latitude of the point.

        Returns:
            float: Raster value at the specified point.
        """
        with rasterio.open(rasterio.MemoryFile().name, 'w+', **metadata) as src:
            src.write(data, 1)
            row, col = src.index(lon, lat)
            return data[row, col]

    def load_shapefile(self, file_path):
        """
        Load a shapefile.

        Args:
            file_path (str): Path to the shapefile.

        Returns:
            geopandas.GeoDataFrame: Loaded shapefile.
        """
        return gpd.read_file(file_path)

    def calculate_zonal_statistics(self, raster_path, vector_path, stats=['mean', 'max', 'min', 'count']):
        """
        Calculate zonal statistics for a raster based on vector geometries.

        Args:
            raster_path (str): Path to the raster file.
            vector_path (str): Path to the vector file.
            stats (list): List of statistics to calculate.

        Returns:
            list: List of dictionaries containing zonal statistics.
        """
        vector = gpd.read_file(vector_path)
        return zonal_stats(vector, raster_path, stats=stats)

    def extract_by_mask(self, raster_path, mask_path, output_path):
        """
        Extract raster data by a vector mask.

        Args:
            raster_path (str): Path to the input raster.
            mask_path (str): Path to the vector mask.
            output_path (str): Path to save the output raster.
        """
        with rasterio.open(raster_path) as src:
            mask = gpd.read_file(mask_path)
            out_image, out_transform = rasterio.mask.mask(src, mask.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

    def rasterize_vector(self, vector_path, reference_raster_path, output_path, attribute):
        """
        Rasterize a vector file based on a reference raster.

        Args:
            vector_path (str): Path to the input vector file.
            reference_raster_path (str): Path to the reference raster file.
            output_path (str): Path to save the output raster.
            attribute (str): Attribute from the vector file to burn into the raster.
        """
        with rasterio.open(reference_raster_path) as ref:
            meta = ref.meta.copy()
            vector = gpd.read_file(vector_path)
            shapes = ((geom, value) for geom, value in zip(vector.geometry, vector[attribute]))
            
            with rasterio.open(output_path, 'w+', **meta) as out:
                out_arr = out.read(1)
                burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                out.write_band(1, burned)

    def create_hillshade(self, dem_path, output_path, azimuth=315, altitude=45):
        """
        Create a hillshade from a Digital Elevation Model (DEM).

        Args:
            dem_path (str): Path to the input DEM file.
            output_path (str): Path to save the output hillshade.
            azimuth (float): Azimuth angle of the light source (0-360).
            altitude (float): Altitude angle of the light source (0-90).
        """
        with rasterio.open(dem_path) as src:
            elevation = src.read(1)
            metadata = src.meta.copy()
            
        x, y = np.gradient(elevation)
        slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.
        altituderad = altitude * np.pi / 180.
        
        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(azimuthrad - aspect)
        
        metadata.update(dtype=rasterio.uint8, count=1)
        
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(shaded * 255, 1)

def process_chunk(args):
    chunk, classifier = args
    return [classifier(lat, lon) for lat, lon in zip(chunk['Latitude'], chunk['Longitude'])]

if __name__ == "__main__":
    # Example usage
    handler = GeoDataHandler()
    
    # Load and query a GeoTIFF
    raster_data, metadata = handler.load_geotiff('path/to/raster.tif')
    value = handler.query_raster_value(raster_data, metadata, -122.4194, 37.7749)
    print(f"Raster value at (-122.4194, 37.7749): {value}")
    
    # Convert NetCDF to GeoTIFF
    handler.netcdf_to_geotiff('path/to/input.nc', 'path/to/output.tif', 'temperature')
    
    # Calculate zonal statistics
    stats = handler.calculate_zonal_statistics('path/to/raster.tif', 'path/to/zones.shp')
    print("Zonal statistics:", stats)