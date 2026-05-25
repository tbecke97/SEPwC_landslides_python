"""
Calculate hazard risk of probability for landslides
"""
# pylint: disable=line-too-long
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import rioxarray
import xarray as xr
from xrspatial import slope as xr_slope
from xrspatial import proximity as xr_proximity
# Imported explicitly to satisfy test suite script requirements
import rasterio  # pylint: disable=unused-import
from rasterio import features  # pylint: disable=unused-import


def extract_values_from_raster(da, shapes):
    """
    Extracts pixel values from a DataArray at the locations of vector shapes.
    Handles both Points and Polygons by using the centroid.
    """
    # Ensure shapes match the raster projection
    shapes = shapes.to_crs(da.rio.crs)

    # Use .centroid to get a single point regardless of geometry type
    values = [
        da.sel(x=s.centroid.x, y=s.centroid.y, method="nearest").values.item()
        for s in shapes.geometry
    ]

    return np.array(values)


def make_classifier(x, y, verbose=False):
    """
    Trains a Random Forest classifier.
    Returns the trained model to satisfy pytest requirements.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, y)

    if verbose:
        # Quick internal check
        predictions = model.predict(x)
        acc = accuracy_score(y, predictions)
        print(f"Internal Training Accuracy: {acc * 100:.2f}%")

    return model


# Flattening the landscape architecture requires passing all aligned
# terrain data layers simultaneously
# pylint: disable=too-many-arguments, too-many-positional-arguments
def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    """
    Uses the trained classifier to predict probability for every pixel.
    """
    # Flatten all layers into a table (matches training dataframe order)
    data = {
        'elev': topo.isel(band=0).values.flatten(),
        'Geol': geo.isel(band=0).values.flatten(),
        'LC': lc.isel(band=0).values.flatten(),
        'slope': slope.values.flatten(),
        'fault': dist_fault.isel(band=0).values.flatten()
    }

    df_map = pd.DataFrame(data)

    # Get probability of class '1' (landslide)
    probs = classifier.predict_proba(df_map)[:, 1]

    # Reshape back to 2D
    return probs.reshape(topo.isel(band=0).shape)


# Spatial ML data frames
# require passing all raster and vector layers simultaneously
# pylint: disable=too-many-arguments, too-many-positional-arguments
def create_dataframe(topo, geo, lc, dist_fault, slope, shapes, landslide_label):
    """
    Combines spatial layers into a DataFrame with column names matching
    the test suite schema (elev, fault, slope, LC, Geol, ls).
    """
    def get_layer(da):
        return da.isel(band=0) if 'band' in da.dims else da

    data = {
        'elev': extract_values_from_raster(get_layer(topo), shapes),
        'Geol': extract_values_from_raster(get_layer(geo), shapes),
        'LC': extract_values_from_raster(get_layer(lc), shapes),
        'slope': extract_values_from_raster(get_layer(slope), shapes),
        'fault': extract_values_from_raster(get_layer(dist_fault), shapes),
        'ls': landslide_label  # This is the target class
    }

    return pd.DataFrame(data)


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


# Pipeline coordinates data loading, syncing, feature engineering, and training variables
# pylint: disable=too-many-locals
def main(args_list=None):
    """
    Main orchestration function running the end-to-end ML pipeline.
    """

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

    # topo provides template for geo and lc
    topo = rioxarray.open_rasterio(args.topography)

    geo = reproject_to_match(rioxarray.open_rasterio(args.geology), topo)
    lc = reproject_to_match(rioxarray.open_rasterio(args.landcover), topo)

    if args.verbose:
        print("SUCCESS: Topography, Geology, and Landcover loaded and aligned.")
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

    # --- Phase 3: Preparing Training Data ---
    if args.verbose:
        print("--- Phase 3: Preparing Training Data ---")

    # Load landslide locations
    landslides = gpd.read_file(args.landslides)

    # 2. Create 'Negative' samples (Non-landslide areas)
    # Create random points where landslides didn't happen to balance the model
    bounds = topo.rio.bounds()
    x_random = np.random.uniform(bounds[0], bounds[2], len(landslides))
    y_random = np.random.uniform(bounds[1], bounds[3], len(landslides))
    non_landslides = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_random, y_random),
        crs=topo.rio.crs
    )

    # 3. Build the DataFrames
    df_pos = create_dataframe(topo, geo, lc, dist_fault, slope, landslides, 1)
    df_neg = create_dataframe(topo, geo, lc, dist_fault, slope, non_landslides, 0)

    # 4. Combine into one master training set
    training_data = pd.concat([df_pos, df_neg])

    if args.verbose:
        print(f"SUCCESS: Created training set with {len(training_data)} rows.")
        print(training_data.head())  # Show the first 5 rows to verify

    # --- Phase 4: Training & Prediction ---
    if args.verbose:
        print("--- Phase 4: Training Classifier ---")

    x_train = training_data.drop(columns=['ls'])
    y_train = training_data['ls']

    # Train the model
    classifier = make_classifier(x_train, y_train, verbose=args.verbose)

    # Generate the probability grid
    prob_array = make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier)

    # --- Phase 5: Saving Output ---
    # Convert the numpy array back into an xarray DataArray to save as TIF
    output_da = xr.DataArray(
        prob_array.astype(np.float32),
        coords=topo.isel(band=0).coords,
        dims=topo.isel(band=0).dims,
        name="probability"
    )

    output_da.rio.to_raster(args.output)

    if args.verbose:
        print(f"SUCCESS: Risk map saved to {args.output}")


if __name__ == '__main__':
    main()
    