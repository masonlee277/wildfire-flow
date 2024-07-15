"""Estimate fire size (acres) and duration (days) from weather and location data."""

import argparse
import os
import time
import sys
import logging

import numpy as np
import elapid as ela
import rasterio as rio
import geopandas as gpd

# file defaults
ROUTINE, _ = os.path.splitext(os.path.basename(__file__))
SCRIPT_DIR, _ = os.path.split(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s %(levelname)s %(name)s [%(funcName)s] | %(message)s"),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

#def add_parser(mainparser: argparse.ArgumentParser):
def parse_args():
    """Reads command line arguments"""
    #parser = mainparser.add_parser(
    #    ROUTINE, help=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    #)
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument(
        "--erc",
        metavar="erc_file",
        required=True,
        type=str,
        help="ERC raster path.",
    )
    parser.add_argument(
        "--bi",
        metavar="bi_file",
        required=True,
        type=str,
        help="Burn index raster path",
    )
    parser.add_argument(
        "--wui",
        metavar="wui_proximity_file",
        required=True,
        type=str,
        help="WUI proximity raster path",
    )
    parser.add_argument(
        "--pyrome",
        metavar="pyrome_file",
        required=True,
        type=str,
        help="Pyrome raster path",
    )
    parser.add_argument(
        "--doy",
        metavar="day_of_year",
        required=True,
        type=int,
        help="day of year (in range 1-366)",
    )
    parser.add_argument(
        "-o",
        metavar="output_file",
        required=True,
        nargs=2,
        type=str,
        help="Output paths. Must pass 2 (Fire size, duration).",
    )

    #parser.set_defaults(run_script=main)
    return parser.parse_args()


def get_lon(row):
    return float(row.geometry.centroid.x)


def get_lat(row):
    return float(row.geometry.centroid.y)


#def main(args):
def main():
    """The main recipe for firesizer"""

    args = parse_args()
    start = time.time()
    logger.info(f"Starting {ROUTINE}")

    # load the models
    logger.info("Loading models")
    size_paths = [os.path.join(SCRIPT_DIR, f"fire-size-model-cv{idx}.ela") for idx in range(5)]
    duration_paths = [os.path.join(SCRIPT_DIR, f"fire-duration-model-cv{idx}.ela") for idx in range(5)]

    size_models = [ela.load_object(path) for path in size_paths]
    duration_models = [ela.load_object(path) for path in duration_paths]

    # read the input raster data
    logger.info("Reading data")
    with rio.open(args.erc) as e, rio.open(args.bi) as b, rio.open(args.wui) as w, rio.open(args.pyrome) as p:
        # get the valid pixel mask from the erc data
        mask = e.read_masks(1)
        valid = mask == 255
        rows, cols = np.where(valid)
        xy = np.zeros((len(rows), 2))
        for i in range(len(rows)):
            xy[i] = e.xy(rows[i], cols[i])
        
        # convert the pixel locations to point geometries
        pts = gpd.GeoSeries(gpd.points_from_xy(xy[:,0], xy[:,1], crs=e.crs))

        # get the output raster profile
        profile = e.profile.copy()

        # read the raw data
        erc = e.read(1)
        bi = b.read(1)
        wui = w.read(1)
        pyrome = p.read(1)

    logger.info("Formatting data")

    # index the data to just the valid pixel locations
    erc = erc[valid]
    bi = bi[valid]
    wui = wui[valid]
    pyrome = pyrome[valid]

    # convert to a data frame
    df = {
        "ERC": erc,
        "BI": bi,
        "WUI proximity": wui,
        "Pyrome": pyrome
    }
    gdf = gpd.GeoDataFrame(df, geometry=pts)

    # set the x/y attributes in lat/lon
    gdf['x'] = gdf.to_crs('EPSG:4326').apply(lambda row: get_lon(row), axis=1)
    gdf['y'] = gdf.to_crs('EPSG:4326').apply(lambda row: get_lat(row), axis=1)

    # set the doy attribute
    gdf['doy'] = args.doy * np.ones(len(gdf))

    # format the xdata
    xcolumns = ["doy", "x", "y", "BI", "ERC", "WUI proximity", "Pyrome"]
    x = gdf[xcolumns]

    # generate the predictions
    logger.info("Generating predictions")
    n_pixels = len(gdf)
    n_models = len(size_models)
    size_preds = np.zeros((n_pixels, n_models), dtype=profile['dtype'])
    duration_preds = np.zeros_like(size_preds)
    for idx, (size_model, duration_model) in enumerate(zip(size_models, duration_models)):
        size_preds[:, idx] = size_model.predict(x)
        duration_preds[:, idx] = duration_model.predict(x)

    # compute a mean ensemble
    size_pred = np.nanmean(size_preds, axis=1)
    duration_pred = np.nanmean(duration_preds, axis=1)

    # convert from 1d vectors to 2d arrays
    size_array = np.zeros((profile["height"],  profile["width"]), dtype=profile["dtype"]) + profile["nodata"]
    duration_array = np.zeros((profile["height"],  profile["width"]), dtype=profile["dtype"]) + profile["nodata"]
    
    size_array[valid] = size_pred
    duration_array[valid] = duration_pred

    # write the output rasters
    logger.info("Writing rasters")
    size_path, duration_path = args.o
    with rio.open(size_path, 'w', **profile) as s:
        s.write(size_array, 1)
        s.set_band_description(1, "Fire size (acres)")

    with rio.open(duration_path, 'w', **profile) as d:
        d.write(duration_array, 1)
        d.set_band_description(1, "Fire duration (days)")

    # wrap up
    end = time.time()
    duration = end - start
    output_paths = " | ".join(args.o)
    logger.info(f"Finished {ROUTINE}")
    logger.info(f"Please see output files: {output_paths}")
    logger.info(f"Time elapsed: {duration:0.3f} seconds")


if __name__ == "__main__":
    main()
