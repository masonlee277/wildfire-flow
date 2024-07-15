import os
import uuid
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json 
from tqdm import tqdm
import glob
import imageio 
import cv2

from src.data.geo_data_handler import GeoDataHandler
from src.data.preprocessor import WildfireDataProcessor
from src.models.normalizing_flow import ConditionalNormalizingFlow
from src.models.cnf_utils import CNFUtils, create_fire_size_range

class WildfireInferenceEngine:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join('wildfire-ignition-generator/data/outputs', uuid.uuid4().hex[:6])
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_config()
        print(f'Files will be saved to: {self.output_dir}')

        self.geo_handler = GeoDataHandler(self.config['data_dir'])
        self.processor = WildfireDataProcessor()
        self.processor.load()

        assert self.processor.fit_performed is True, "Load Normalization Statistics in the Pre-Proc Module."

        # self.ignitions_crs = CRS.from_epsg(5070)  # EPSG:5070 for ignitions
        # self.model_crs = CRS.from_epsg(4326)  # EPSG:4326 for the model
        # self.transformer = Transformer.from_crs(self.ignitions_crs, self.model_crs, always_xy=True)


        self.cnf_utils = self.load_cnf_model()

        print('-'*50)
        print('loaded Inference Engine')
        print('-'*50)


    def load_ignitions_csv(self, csv_path, input_crs='utm'):
        """Load and process the ignitions CSV file."""
        ignitions_df = pd.read_csv(csv_path)
        
        # Convert coordinates to WGS84 for internal processing
        lon, lat = self.geo_handler.convert_to_wgs84(ignitions_df.x, ignitions_df.y, from_crs=input_crs)
        
        ignitions_gdf = gpd.GeoDataFrame(
            ignitions_df, 
            geometry=gpd.points_from_xy(lon, lat),
            crs=self.geo_handler.wgs84_crs
        )
        ignitions_gdf['Longitude'], ignitions_gdf['Latitude'] = lon, lat
        return ignitions_gdf

    def prepare_ignitions_for_prediction(self, ignitions_gdf, date):
        """Prepare ignitions data for prediction."""
        daily_df = self.geo_handler.get_daily_dataframe(date)
        merged_df = gpd.sjoin_nearest(ignitions_gdf, daily_df, how="left")
        preprocessed_df = self.processor.transform(merged_df)
        X, _ = self.processor.prepare_for_training(preprocessed_df)
        return X, merged_df

    def predict_fire_sizes(self, X, merged_df):
        """Predict fire sizes for ignitions."""
        predictions_df = self.predict(X)
        final_df = pd.concat([merged_df.reset_index(drop=True), predictions_df], axis=1)
        return final_df

    def create_prescribed_ignitions_datastructure(self, final_df, output_crs='utm'):
        """Create a data structure for prescribed ignitions."""
        prescribed_ignitions = final_df[['icase', 'iwx_band', 'Longitude', 'Latitude', 'stop_above_atotoal', 'stop_at_t']].copy()
        prescribed_ignitions['predicted_50th_percentile'] = final_df['50th_percentile']
        prescribed_ignitions['predicted_95th_percentile'] = final_df['95th_percentile']
        prescribed_ignitions['prob_above_10000'] = final_df['prob_above_10000']
        
        # Convert coordinates back to the desired output CRS
        x, y = self.geo_handler.convert_from_wgs84(prescribed_ignitions.Longitude, prescribed_ignitions.Latitude, to_crs=output_crs)
        prescribed_ignitions['x'] = x
        prescribed_ignitions['y'] = y
        
        return prescribed_ignitions

    def save_prescribed_ignitions(self, prescribed_ignitions, filename):
        """Save the prescribed ignitions data structure."""
        file_path = os.path.join(self.output_dir, f'{filename}.csv')
        prescribed_ignitions.to_csv(file_path, index=False)
        print(f"Prescribed ignitions saved to {file_path}")

    def run_inference_on_ignitions(self, ignitions_csv_path, date, input_crs='utm', output_crs='utm'):
        """Run inference on ignitions CSV."""
        ignitions_gdf = self.load_ignitions_csv(ignitions_csv_path, input_crs)
        X, merged_df = self.prepare_ignitions_for_prediction(ignitions_gdf, date)
        final_df = self.predict_fire_sizes(X, merged_df)
        prescribed_ignitions = self.create_prescribed_ignitions_datastructure(final_df, output_crs)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dataframe(final_df, f'final_df_ignitions_{timestamp}')
        self.save_prescribed_ignitions(prescribed_ignitions, f'prescribed_ignitions_{timestamp}')
        
        self.plot_ignitions_map(final_df, '95th_percentile', 'Fire Size (95th Percentile)')
        self.plot_ignitions_map(final_df, 'prob_above_10000', 'Probability of Fire Size > 10,000 acres')
        
        return final_df, prescribed_ignitions

    def plot_ignitions_map(self, df, column, title):
        """Plot a map of ignitions colored by the specified column."""
        fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
        
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs=self.model_crs)
        
        vmin, vmax = df[column].min(), df[column].max()
        cmap = 'viridis' if 'percentile' in column else 'RdYlBu_r'
        
        gdf.plot(ax=ax, column=column, cmap=cmap, legend=True, 
                 legend_kwds={'label': title}, markersize=10, alpha=0.7, 
                 edgecolor='none', vmin=vmin, vmax=vmax)
        
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TerrainBackground)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_name = f"{column}_ignitions_map_{timestamp}.png"
        fig_path = os.path.join(self.output_dir, fig_name)
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Ignitions map saved to {fig_path}")


    def load_cnf_model(self):
        def create_model():
            return lambda: ConditionalNormalizingFlow(
                context_size=self.config['context_size'],
                latent_size=self.config['latent_size'],
                num_flow_layers=self.config['num_flow_layers'],
                hidden_units=self.config['hidden_units'],
                hidden_layers=self.config['hidden_layers']
            ).model

        return CNFUtils.from_local(
            model_dir=self.config['model_dir'],
            model_class=create_model()
        )

    def load_data(self, year):
        self.geo_handler.load_netcdf_data("bi", year)
        self.geo_handler.load_netcdf_data("erc", year)
        self.geo_handler.load_wui_data(self.config['wui_data_path'])
        print('Data loaded successfully')

    def process_date(self, date):
        daily_df = self.geo_handler.get_daily_dataframe(date)
        preprocessed_df = self.processor.transform(daily_df)
        
        filtered_daily_df = daily_df[daily_df['original_index'].isin(preprocessed_df['original_index'])]
        
        sorted_preprocessed_df = preprocessed_df.sort_values(by='original_index', ascending=True).reset_index(drop=True)
        sorted_filtered_daily_df = filtered_daily_df.sort_values(by='original_index', ascending=True).reset_index(drop=True)
        
        sorted_preprocessed_df = sorted_preprocessed_df.drop(columns=['original_index'])
        
        X, y = self.processor.prepare_for_training(sorted_preprocessed_df)
        
        return X, sorted_filtered_daily_df

    def predict(self, X, percentiles=None, thresholds=None):
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]
        if thresholds is None:
            thresholds = [100, 1000, 10000]

        print(f'Percentiles: {percentiles}')
        fire_size_range = create_fire_size_range().to(self.cnf_utils.device)
        results = []
        
        batch_size = 1000
        num_batches = (len(X) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X))
            
            context_batch = X[start_idx:end_idx]
            
            fire_sizes_log_repeated = fire_size_range.repeat(len(context_batch), 1)
            context_repeated = context_batch.unsqueeze(1).repeat(1, fire_size_range.shape[0], 1)
            
            log_probs = self.cnf_utils.log_prob(fire_sizes_log_repeated.view(-1, 1), context_repeated.view(-1, context_batch.shape[1]))
            log_probs = log_probs.view(len(context_batch), -1)
            
            fire_sizes = np.expm1(fire_size_range.cpu().numpy()).flatten()
            pdf_values = torch.exp(log_probs).cpu().numpy()
            cdf_values = np.cumsum(pdf_values, axis=1)
            cdf_values /= cdf_values[:, -1:]
            
            percentile_values = np.array([fire_sizes[np.searchsorted(cdf, p/100)] for cdf in cdf_values for p in percentiles]).reshape(-1, len(percentiles))
            
            probs_above_threshold = [1 - cdf_values[:, np.searchsorted(fire_sizes, threshold)] for threshold in thresholds]
            
            for i in range(len(context_batch)):
                result = {
                    f'{p}th_percentile': percentile_values[i, j] for j, p in enumerate(percentiles)
                }
                result.update({
                    f'prob_above_{threshold}': prob[i] for threshold, prob in zip(thresholds, probs_above_threshold)
                })
                results.append(result)

        return pd.DataFrame(results)

    def generate_final_df(self, date, percentiles=None, thresholds=None):
        X, sorted_filtered_daily_df = self.process_date(date)
        predictions_df = self.predict(X, percentiles, thresholds)
        final_df = pd.concat([sorted_filtered_daily_df.reset_index(drop=True), predictions_df], axis=1)
        return final_df

    def load_dataframe(self, file_path):
        """Load a DataFrame from a pickle file."""
        return pd.read_pickle(file_path)

    def generate_binary_map(self, final_df, threshold_size, percentile=99.9, load_from_file=None):
        if load_from_file:
            final_df = self.load_dataframe(load_from_file)
            print(final_df.head)
        
        X, sorted_filtered_daily_df = self.process_date(final_df['Ignition date'].iloc[0])
        
        batch_size = 1000
        num_batches = (len(X) + batch_size - 1) // batch_size
        
        print('binary map predictions')
        percentile_values = []
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X))
            
            context_batch = X[start_idx:end_idx]
            
            percentile_batch = self.cnf_utils.calculate_percentiles_batch(context_batch, percentile)
            percentile_values.extend(percentile_batch)
        
        final_df[f'{percentile}th_percentile'] = percentile_values
        final_df['above_threshold'] = final_df[f'{percentile}th_percentile'] > threshold_size
        
        return final_df

    def run_inference_with_binary_map(self, date, threshold_size, percentile=99.9, load_from_file=None):
        if load_from_file:
            final_df = self.load_dataframe(load_from_file)
        else:
            final_df = self.generate_final_df(date)
            try:
                self.save_dataframe(final_df, f'final_df_{date.strftime("%Y%m%d")}')
            except:
                print('Saving Dataframe Failed')
        
        final_df_with_binary = self.generate_binary_map(final_df, threshold_size, percentile)
        
        try:
            self.save_dataframe(final_df_with_binary, f'final_df_with_binary_{date.strftime("%Y%m%d")}')
        except:
            print('Saving Dataframe Failed')
        
        try:
            self.plot_binary_map(final_df_with_binary, threshold_size, percentile)
        except Exception as e:
            print(f'Plotting Binary Map Failed: {str(e)}')
        
        return final_df_with_binary

    def plot_binary_map(self, df, threshold_size, percentile=99.9):
        pyrome_gdf = gpd.read_file(self.config['pyrome_shapefile_path'])
        pyrome_gdf = pyrome_gdf.to_crs(epsg=4326)
        
        unique_pyromes = df['Pyrome'].unique()
        pyrome_gdf = pyrome_gdf[pyrome_gdf['PYROME'].isin(unique_pyromes)]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
        
        pyrome_gdf.plot(ax=ax, color='none', edgecolor='k', linewidth=0.5, alpha=0.5)
        
        scatter = gdf.plot(ax=ax, column='above_threshold', cmap='RdYlBu_r', 
                        legend=True, legend_kwds={'label': f'Above {threshold_size} acres at {percentile}th percentile'},
                        markersize=10, alpha=0.7, edgecolor='none', categorical=True)
        
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TerrainBackground)
        except Exception as e:
            print(f"Failed to add basemap: {e}")
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        ignition_date = df['Ignition date'].iloc[0]
        ax.set_title(f'Binary Map: Fire Size > {threshold_size} acres at {percentile}th percentile\nIgnition Date: {ignition_date}', fontsize=16, fontweight='bold')
        
        x, y, arrow_length = 0.05, 0.95, 0.1
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)
        
        def add_scalebar(ax, length, location=(0.85, 0.05), linewidth=3):
            sbx, sby = location
            ax.plot([sbx, sbx + length], [sby, sby], transform=ax.transAxes, color='k', linewidth=linewidth)
            ax.plot([sbx, sbx], [sby, sby-0.01], transform=ax.transAxes, color='k', linewidth=linewidth)
            ax.plot([sbx + length, sbx + length], [sby, sby-0.01], transform=ax.transAxes, color='k', linewidth=linewidth)
            ax.text(sbx + length/2, sby-0.03, f'{int(length * 111)}km', transform=ax.transAxes, 
                    ha='center', va='top', fontsize=10, fontweight='bold')

        add_scalebar(ax, 0.1)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        plt.tight_layout()
        
        fig_name = f"binary_map_{threshold_size}acres_{percentile}percentile_{timestamp}.png"
        fig_path = os.path.join(self.output_dir, fig_name)
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Binary map saved to {fig_path}")


    def save_config(self):
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {config_path}")

    def save_dataframe(self, df, filename):
        file_path = os.path.join(self.output_dir, f'{filename}.pkl')
        df.to_pickle(file_path)
        print(f"DataFrame saved as pickle to {file_path}")

    def plot_map(self, df, column, plot_type):
        pyrome_gdf = gpd.read_file(self.config['pyrome_shapefile_path'])
        pyrome_gdf = pyrome_gdf.to_crs(epsg=4326)
        
        unique_pyromes = df['Pyrome'].unique()
        pyrome_gdf = pyrome_gdf[pyrome_gdf['PYROME'].isin(unique_pyromes)]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
        
        pyrome_gdf.plot(ax=ax, color='none', edgecolor='k', linewidth=0.5, alpha=0.5)
        
        vmin, vmax = df[column].min(), df[column].max()
        
        if plot_type == 'percentile':
            cmap = 'viridis'
            label = f'{column} Fire Size (acres)'
        elif plot_type == 'threshold':
            cmap = 'RdYlBu_r'
            label = f'Probability of fire size > {column.split("_")[-1]} acres'
        
        scatter = gdf.plot(ax=ax, column=column, cmap=cmap, 
                           legend=True, legend_kwds={'label': label},
                           markersize=10, alpha=0.7, edgecolor='none', vmin=vmin, vmax=vmax)
        
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TerrainBackground)
        except Exception as e:
            print(f"Failed to add basemap: {e}")
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        ignition_date = df['Ignition date'].iloc[0]
        ax.set_title(f'{column} Map\nIgnition Date: {ignition_date}', fontsize=16, fontweight='bold')
        
        x, y, arrow_length = 0.05, 0.95, 0.1
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)
        
        def add_scalebar(ax, length, location=(0.85, 0.05), linewidth=3):
            sbx, sby = location
            ax.plot([sbx, sbx + length], [sby, sby], transform=ax.transAxes, color='k', linewidth=linewidth)
            ax.plot([sbx, sbx], [sby, sby-0.01], transform=ax.transAxes, color='k', linewidth=linewidth)
            ax.plot([sbx + length, sbx + length], [sby, sby-0.01], transform=ax.transAxes, color='k', linewidth=linewidth)
            ax.text(sbx + length/2, sby-0.03, f'{int(length * 111)}km', transform=ax.transAxes, 
                    ha='center', va='top', fontsize=10, fontweight='bold')

        add_scalebar(ax, 0.1)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        plt.tight_layout()
        
        fig_name = f"{column}_map_{timestamp}.png"
        fig_path = os.path.join(self.output_dir, fig_name)
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Map saved to {fig_path}")

    def create_percentile_gif(self, image_dir, output_filename='percentile_gif.gif', duration=5.0):
        images = []
        image_files = sorted(glob.glob(os.path.join(image_dir, '*th_percentile_map_*.png')))
        
        print(f"Found {len(image_files)} image files.")
        
        # Read the first image to get the target dimensions
        target_image = cv2.imread(image_files[0])
        target_height, target_width = target_image.shape[:2]
        print(f"Target dimensions: {target_width}x{target_height}")
        
        for i, image_file in enumerate(image_files):
            img = cv2.imread(image_file)
            print(f"Image {i+1}: {image_file}")
            print(f"  Original shape: {img.shape}")
            
            # Resize the image if it doesn't match the target dimensions
            if img.shape[:2] != (target_height, target_width):
                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                print(f"  Resized to: {img.shape}")
            else:
                print("  No resizing needed")
            
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            images.append(img)
        
        print("\nFinal image shapes:")
        for i, img in enumerate(images):
            print(f"  Image {i+1}: {img.shape}")
        
        # Calculate the duration for each frame (total duration divided by number of images)
        frame_duration = duration / len(images)
        print(f"\nFrame duration: {frame_duration:.4f} seconds")

        # Save the GIF
        gif_path = os.path.join(self.output_dir, output_filename)
        print(f"\nAttempting to save GIF to: {gif_path}")
        print(f"Number of frames: {len(images)}")
        
        try:
            imageio.mimsave(gif_path, images, duration=frame_duration, loop=0)
            print(f"Percentile GIF saved successfully to {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {str(e)}")
            print("Shapes of all images:")
            for i, img in enumerate(images):
                print(f"  Image {i+1}: {img.shape}")

        # If the GIF creation fails, try saving with PIL
        if not os.path.exists(gif_path):
            try:
                print("Attempting to save GIF using PIL...")
                from PIL import Image
                pil_images = [Image.fromarray(img) for img in images]
                pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=int(frame_duration*1000), loop=0)
                print(f"Percentile GIF saved successfully using PIL to {gif_path}")
            except Exception as e:
                print(f"Error saving GIF with PIL: {str(e)}")

        return gif_path
    def plot_percentile_maps(self, final_df, percentiles):
        for percentile in tqdm(percentiles, desc="Generating percentile plots"):
            column = f'{percentile}th_percentile'
            self.plot_map(final_df, column, 'percentile')



    def run_inference_with_gif(self, date):
        percentiles = list(range(1, 100)) + [99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 99.99]
        final_df = self.generate_final_df(date, percentiles)
        
        try:
            self.save_dataframe(final_df, f'final_df_{date.strftime("%Y%m%d")}')
        except:
            print('Saving Dataframe Failed')
        
        
        try:
            self.plot_percentile_maps(final_df, percentiles)
        except Exception as e:
            print(f'Plotting Percentile Maps Failed: {str(e)}')
        
        try:
            self.create_percentile_gif(self.output_dir)
        except Exception as e:
            print(f'Creating Percentile GIF Failed: {str(e)}')

        return final_df

    def run_inference(self, date):
        final_df = self.generate_final_df(date)
        try:
            self.save_dataframe(final_df, f'final_df_{date.strftime("%Y%m%d")}')
        except:
            print('Saving Dataframe Failed')
            
        try:
            percentile_columns = [col for col in final_df.columns if col.endswith('th_percentile')]
            threshold_columns = [col for col in final_df.columns if col.startswith('prob_above_')]
            
            for col in percentile_columns:
                self.plot_map(final_df, col, 'percentile')
            
            for col in threshold_columns:
                self.plot_map(final_df, col, 'threshold')
        except Exception as e:
            print(f'Plotting Dataframe Failed: {str(e)}')

        return final_df


if __name__ == "__main__":
    config = {
        'data_dir': "wildfire-ignition-generator/data/wildfire_indicators/raw_netcdfs",
        'wui_data_path': 'placeholder_wui.tif',
        'model_dir': 'wildfire-ignition-generator/data/artifacts/run-n6xf1a04-history:v0/local',
        'context_size': 75,
        'latent_size': 1,
        'num_flow_layers': 4,
        'hidden_units': 128,
        'hidden_layers': 3,
        'pyrome_shapefile_path': 'wildfire-ignition-generator/data/pyrome_shp/Pyromes_CONUS_20200206.shp'
    }

    engine = WildfireInferenceEngine(config)
    engine.load_data(2023)
    date = datetime(2023, 7, 1)
    final_df = engine.run_inference(date)
    print(final_df.head())