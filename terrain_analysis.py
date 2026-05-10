"""
Calculate hazard risk of probability for landslides
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import rioxarray
import rasterio
import xarray as xr
from xrspatial import slope as xr_slope
from xrspatial import proximity as xr_proximity

def extract_values_from_raster(da, shapes):
    return
    

def make_classifier(x, y, verbose=False):
    return


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    return


def create_dataframe(topo, geo, lc, dist_fault, slope, shapes, landslide_label):
    return

def reproject_to_match(in_raster, template_raster):
    return


def calculate_distance_to_faults(fault_shapefile, template_raster):
    return


def main(args_list=None):
    """
    Main entry point for the Landslide Susceptibility Mapping script.
    """
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(description="Calculate landslide susceptibility maps.")
    
    # Flags for inputs and verbosity
    parser.add_argument('--topography', required=True, help="Path to topography raster")
    parser.add_argument('--geology', required=True, help="Path to geology raster")
    parser.add_argument('-v', '--verbose', action='store_true', help="Provide progress updates")

    # Positional arguments
    parser.add_argument("landslides", help="Path to landslide shapefile")
    parser.add_argument("output", help="Path for output probability raster")

    # 2. Parse the arguments
    args = parser.parse_args(args_list)

    # 3. Progress Update (Criterion 1: Gives updates to the user)
    if args.verbose:
        print("--- Landslide Analysis Started ---")
        print(f"Using topography: {args.topography}")

    return args
