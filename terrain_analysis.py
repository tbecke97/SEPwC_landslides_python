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
    
    """
    Ensures the input raster matches the CRS, resolution, and extent 
    of the template raster. Essential for pixel-wise ML analysis.
    """
    return in_raster.rio.reproject_match(template_raster)


def calculate_distance_to_faults(fault_shapefile, template_raster):
    """
    Computes the distance from every pixel to the nearest fault line using xrspatial
    """
    # Load and reproject vector data to match our raster grid
    faults = gpd.read_file(fault_shapefile)
    faults = faults.to_crs(template_raster.rio.crs)
   
    # Rasterize: Create a 2D float grid where faults are 1 and background is NaN
    from rasterio import features
    mask = features.rasterize(
        [(shape, 1) for shape in faults.geometry],
        out_shape=template_raster.shape[-2:],
        transform=template_raster.rio.transform(),
        fill=np.nan,
        all_touched=True,
        dtype='float32'
    )
    
    # Convert to DataArray using 2D coordinates to match mask dimensions
    mask_da = xr.DataArray(
        mask, 
        coords=template_raster.isel(band=0).coords, 
        dims=template_raster.isel(band=0).dims
        )
    
    # Generate proximity map (distance in coordinate units)
    dist_fault = xr_proximity(mask_da)
    
    return dist_fault.expand_dims(dim="band", axis=0)
    




def main(args_list=None):
    parser = argparse.ArgumentParser(
        prog="Landslide hazard using ML",
        description="Calculate landslide hazards using machine learning"
    )
    parser.add_argument('--topography', required=True, help="topographic raster file")
    parser.add_argument('--geology', required=True, help="geology raster file")
    parser.add_argument('--landcover', required=True, help="landcover raster file")
    parser.add_argument('--faults', required=True, help="fault location shapefile")
    parser.add_argument("landslides", help="landslide location shapefile")
    parser.add_argument("output", help="output probability raster file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print progress")
    
    args = parser.parse_args(args_list)

    if args.verbose:
        print("--- Phase 1: Loading & Syncing Data ---")
        
        """
       topo provides template for geo and lc
        """
        
    topo = rioxarray.open_rasterio(args.topography)
    
    geo = reproject_to_match(rioxarray.open_rasterio(args.geology), topo)
    lc = reproject_to_match(rioxarray.open_rasterio(args.landcover), topo)
    
    if args.verbose:
        print(f"SUCCESS: Topography, Geology, and Landcover loaded and aligned.")
        print(f"Common Shape: {topo.shape}")
        
    # --- Phase 2: Generating Terrain Features ---
    if args.verbose:
        print("--- Phase 2: Generating Terrain Features ---")
        
    # Generate Slope: xrspatial requires a 2D input, so we select the first band
    slope = xr_slope(topo.sel(band=1))
    
    # Generate Fault Proximity: Uses the custom function defined above
    dist_fault = calculate_distance_to_faults(args.faults, topo)
        
    if args.verbose:
        print("SUCCESS: Terrain features generated.")
        
if __name__ == '__main__':
    main()
    
    
    
    
    