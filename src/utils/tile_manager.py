"""
TileManager: A comprehensive class for managing geospatial tiles.

This class provides methods for creating, managing, and processing geospatial tiles.
It supports tiling based on geographic boundaries, specific pyromes, and allows for
potential parallelization of operations.

Key features:
- Create tiles based on geographic boundaries
- Filter tiles based on specific pyromes
- Generate tiles for a given area of interest
- Process tiles in parallel
- Query which tile a given coordinate belongs to
- Iterate over tiles
- Save and load tile configurations

Dependencies:
- numpy
- geopandas
- shapely
- multiprocessing
- json

Usage:
    manager = TileManager(tile_size=1.0)
    manager.create_tiles(bounds=(-125, 25, -65, 50))
    manager.filter_tiles_by_pyromes([1, 2, 3])
    for tile in manager.iter_tiles():
        process_tile(tile)

Note: Ensure all required libraries are installed and properly configured.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
from multiprocessing import Pool
import json
import os

class Tile:
    def __init__(self, x_min, y_min, x_max, y_max, pyrome=None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.pyrome = pyrome

    @property
    def bounds(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def geometry(self):
        return box(*self.bounds)

    def __repr__(self):
        return f"Tile(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max}, pyrome={self.pyrome})"

class TileManager:
    def __init__(self, tile_size=1.0):
        self.tile_size = tile_size
        self.tiles = []
        self.pyrome_gdf = None

    def create_tiles(self, bounds):
        """
        Create tiles based on geographic boundaries.

        Args:
            bounds (tuple): (min_x, min_y, max_x, max_y) of the area to tile
        """
        x_min, y_min, x_max, y_max = bounds
        x_tiles = int((x_max - x_min) / self.tile_size)
        y_tiles = int((y_max - y_min) / self.tile_size)

        self.tiles = []
        for i in range(x_tiles):
            for j in range(y_tiles):
                tile_x_min = x_min + i * self.tile_size
                tile_y_min = y_min + j * self.tile_size
                tile_x_max = tile_x_min + self.tile_size
                tile_y_max = tile_y_min + self.tile_size
                self.tiles.append(Tile(tile_x_min, tile_y_min, tile_x_max, tile_y_max))

    def load_pyrome_shapefile(self, shapefile_path):
        """
        Load pyrome shapefile for filtering tiles.

        Args:
            shapefile_path (str): Path to the pyrome shapefile
        """
        self.pyrome_gdf = gpd.read_file(shapefile_path)

    def filter_tiles_by_pyromes(self, pyrome_ids):
        """
        Filter tiles based on specific pyrome IDs.

        Args:
            pyrome_ids (list): List of pyrome IDs to include
        """
        if self.pyrome_gdf is None:
            raise ValueError("Pyrome shapefile not loaded. Call load_pyrome_shapefile first.")

        filtered_tiles = []
        for tile in self.tiles:
            tile_geom = tile.geometry
            intersecting_pyromes = self.pyrome_gdf[self.pyrome_gdf.intersects(tile_geom)]
            if not intersecting_pyromes.empty:
                pyrome_id = intersecting_pyromes.iloc[0]['PYROME']
                if pyrome_id in pyrome_ids:
                    tile.pyrome = pyrome_id
                    filtered_tiles.append(tile)

        self.tiles = filtered_tiles

    def get_tile(self, lon, lat):
        """
        Get the tile that contains the given coordinates.

        Args:
            lon (float): Longitude
            lat (float): Latitude

        Returns:
            Tile: The tile containing the coordinates, or None if not found
        """
        point = Point(lon, lat)
        for tile in self.tiles:
            if tile.geometry.contains(point):
                return tile
        return None

    def iter_tiles(self):
        """
        Iterator for tiles.

        Yields:
            Tile: Next tile in the collection
        """
        for tile in self.tiles:
            yield tile

    def process_tiles_parallel(self, process_func, num_processes=None):
        """
        Process tiles in parallel.

        Args:
            process_func (callable): Function to process each tile
            num_processes (int, optional): Number of processes to use. Defaults to number of CPUs.

        Returns:
            list: Results of processing each tile
        """
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, self.tiles)
        return results

    def save_configuration(self, file_path):
        """
        Save the current tile configuration to a file.

        Args:
            file_path (str): Path to save the configuration file
        """
        config = {
            'tile_size': self.tile_size,
            'tiles': [
                {
                    'x_min': tile.x_min,
                    'y_min': tile.y_min,
                    'x_max': tile.x_max,
                    'y_max': tile.y_max,
                    'pyrome': tile.pyrome
                } for tile in self.tiles
            ]
        }
        with open(file_path, 'w') as f:
            json.dump(config, f)

    def load_configuration(self, file_path):
        """
        Load a tile configuration from a file.

        Args:
            file_path (str): Path to the configuration file
        """
        with open(file_path, 'r') as f:
            config = json.load(f)

        self.tile_size = config['tile_size']
        self.tiles = [
            Tile(
                tile['x_min'],
                tile['y_min'],
                tile['x_max'],
                tile['y_max'],
                tile['pyrome']
            ) for tile in config['tiles']
        ]

    def get_tiles_in_bbox(self, bbox):
        """
        Get tiles that intersect with a given bounding box.

        Args:
            bbox (tuple): (min_x, min_y, max_x, max_y) of the bounding box

        Returns:
            list: List of tiles intersecting the bounding box
        """
        bbox_geom = box(*bbox)
        return [tile for tile in self.tiles if tile.geometry.intersects(bbox_geom)]

    def get_tiles_by_pyrome(self, pyrome_id):
        """
        Get all tiles associated with a specific pyrome.

        Args:
            pyrome_id (int): ID of the pyrome

        Returns:
            list: List of tiles associated with the pyrome
        """
        return [tile for tile in self.tiles if tile.pyrome == pyrome_id]

    def create_tile_index(self):
        """
        Create a spatial index for faster tile lookups.

        This method creates a simple grid-based spatial index.
        """
        x_min = min(tile.x_min for tile in self.tiles)
        y_min = min(tile.y_min for tile in self.tiles)
        x_max = max(tile.x_max for tile in self.tiles)
        y_max = max(tile.y_max for tile in self.tiles)

        nx = int((x_max - x_min) / self.tile_size)
        ny = int((y_max - y_min) / self.tile_size)

        self.tile_index = [[[] for _ in range(ny)] for _ in range(nx)]

        for tile in self.tiles:
            i = int((tile.x_min - x_min) / self.tile_size)
            j = int((tile.y_min - y_min) / self.tile_size)
            self.tile_index[i][j].append(tile)

    def get_tile_from_index(self, lon, lat):
        """
        Get a tile using the spatial index.

        Args:
            lon (float): Longitude
            lat (float): Latitude

        Returns:
            Tile: The tile containing the coordinates, or None if not found
        """
        if not hasattr(self, 'tile_index'):
            self.create_tile_index()

        x_min = min(tile.x_min for tile in self.tiles)
        y_min = min(tile.y_min for tile in self.tiles)

        i = int((lon - x_min) / self.tile_size)
        j = int((lat - y_min) / self.tile_size)

        if 0 <= i < len(self.tile_index) and 0 <= j < len(self.tile_index[0]):
            point = Point(lon, lat)
            for tile in self.tile_index[i][j]:
                if tile.geometry.contains(point):
                    return tile
        return None

    def generate_tile_geojson(self, output_path):
        """
        Generate a GeoJSON file of the tiles.

        Args:
            output_path (str): Path to save the GeoJSON file
        """
        features = []
        for tile in self.tiles:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(tile.geometry.exterior.coords)]
                },
                "properties": {
                    "pyrome": tile.pyrome
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f)

    def merge_adjacent_tiles(self, max_area=None):
        """
        Merge adjacent tiles with the same pyrome ID.

        Args:
            max_area (float, optional): Maximum area of merged tiles. Defaults to None.

        This method will merge adjacent tiles that have the same pyrome ID,
        potentially reducing the total number of tiles and simplifying processing.
        """
        merged_tiles = []
        processed = set()

        for i, tile in enumerate(self.tiles):
            if i in processed:
                continue

            merged_geom = tile.geometry
            merged_pyrome = tile.pyrome
            to_merge = [j for j, other in enumerate(self.tiles) if j != i and
                        other.pyrome == merged_pyrome and merged_geom.touches(other.geometry)]

            while to_merge:
                j = to_merge.pop(0)
                if j in processed:
                    continue
                other_geom = self.tiles[j].geometry
                new_geom = merged_geom.union(other_geom)
                if max_area is None or new_geom.area <= max_area:
                    merged_geom = new_geom
                    processed.add(j)
                    to_merge.extend([k for k, another in enumerate(self.tiles) if k != j and k not in processed and
                                     another.pyrome == merged_pyrome and merged_geom.touches(another.geometry)])

            processed.add(i)
            bounds = merged_geom.bounds
            merged_tiles.append(Tile(bounds[0], bounds[1], bounds[2], bounds[3], pyrome=merged_pyrome))

        self.tiles = merged_tiles

    def split_large_tiles(self, max_area):
        """
        Split tiles larger than a specified maximum area.

        Args:
            max_area (float): Maximum allowed area for a tile
        """
        new_tiles = []
        for tile in self.tiles:
            if tile.geometry.area > max_area:
                # Calculate how many times to split in each direction
                split_factor = int(np.ceil(np.sqrt(tile.geometry.area / max_area)))
                dx = (tile.x_max - tile.x_min) / split_factor
                dy = (tile.y_max - tile.y_min) / split_factor

                for i in range(split_factor):
                    for j in range(split_factor):
                        new_x_min = tile.x_min + i * dx
                        new_y_min = tile.y_min + j * dy
                        new_x_max = new_x_min + dx
                        new_y_max = new_y_min + dy
                        new_tiles.append(Tile(new_x_min, new_y_min, new_x_max, new_y_max, pyrome=tile.pyrome))
            else:
                new_tiles.append(tile)

        self.tiles = new_tiles

    def optimize_tile_layout(self, max_area=None, min_area=None):
        """
        Optimize the tile layout by merging small tiles and splitting large ones.

        Args:
            max_area (float, optional): Maximum allowed area for a tile. Defaults to None.
            min_area (float, optional): Minimum allowed area for a tile. Defaults to None.
        """
        if min_area is not None:
            self.merge_adjacent_tiles(max_area=max_area)
        if max_area is not None:
            self.split_large_tiles(max_area)

    def get_tile_statistics(self):
        """
        Get statistics about the current tile configuration.

        Returns:
            dict: Statistics about the tiles
        """
        areas = [tile.geometry.area for tile in self.tiles]
        pyrome_counts = {}
        for tile in self.tiles:
            pyrome_counts[tile.pyrome] = pyrome_counts.get(tile.pyrome, 0) + 1

        return {
            "num_tiles": len(self.tiles),
            "total_area": sum(areas),
            "min_area": min(areas),
            "max_area": max(areas),
            "mean_area": np.mean(areas),
            "median_area": np.median(areas),
            "pyrome_counts": pyrome_counts
        }

if __name__ == "__main__":
    # Example usage
    manager = TileManager(tile_size=1.0)
    manager.create_tiles(bounds=(-125, 25, -65, 50))
    manager.load_pyrome_shapefile('path/to/pyrome_shapefile.shp')
    manager.filter_tiles_by_pyromes([1, 2, 3])

    # Process tiles in parallel
    def process_tile(tile):
        # Example processing function
        return f"Processed tile: {tile}"

    results = manager.process_tiles_parallel(process_tile)
    print(f"Processed {len(results)} tiles")

    # Save and load configuration
    manager.save_configuration('tile_config.json')
    manager.load_configuration('tile_config.json')

    # Generate GeoJSON of tiles
    manager.generate_tile_geojson('tiles.geojson')

    # Optimize tile layout
    manager.optimize_tile_layout(max_area=2.0, min_area=0.5)

    # Get tile statistics
    stats = manager.get_tile_statistics()
    print("Tile statistics:", stats)