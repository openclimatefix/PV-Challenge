"""
The idea is to extract the NWP data at specific PV site locations

The current format of the NWP data is zarr file which can be loaded into xarray
Dimensions are   (variable: 17, init_time: 5319, step: 37, y: 704, x: 548)

Idea is to reduce to variables
- si10: wind speed
- dswrf: Downward longwave radiation flux - ground
- t:  Air temperature at 1 meter above surface in Kelvin.
- prate: Precipitation rate at the surface in kg/m^2/s.


A file is saved for each location in 'data/locations.csv'
Idea is to reduce x and y to the locations in the csv above.
We want to save eahc site as a serperate 'netcdf'.
Afterwards we can collect these into one big file.
"""

import xarray as xr
import pandas as pd
import os
from tqdm import tqdm

zarr_path = "~/../../mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/zarr/UKV_intermediate_version_3.zarr"

# open the dataset, but not into memory
nwp = xr.open_dataset(
    zarr_path,
    engine="zarr",
    consolidated=True,
    mode="r",
    chunks=None,  # Reading Satellite Zarr benefits from setting chunks='auto'
    # (see issue #456) but 'auto' massively slows down reading NWPs.
)

# select the variables we want
nwp_all = nwp.sel(variable=["si10", "dswrf", "t", "prate"])

# get locations
locations = pd.read_csv("./data/locations.csv")


def extract_one_location(nwp, x_osgb, y_osgb, id):
    """
    Function to extrac one location from a big nwp file

    :param nwp: xarray
    :param x_osgb: x location in OSGB coorindates
    :param y_osgb: y location in OSGB coorindates
    :param id: the id of the site
    """

    # the x and y coordinates are in a 2000 grid = 2km. So lets choose the closest ones
    nwp = nwp.sel(x=((nwp.x > x_osgb - 2000) & (nwp.x < x_osgb + 2000)))
    nwp = nwp.sel(y=((nwp.y > y_osgb - 2000) & (nwp.y < y_osgb + 2000)))

    # COul interpolate but for the moment, lets just take the first value
    nwp = nwp.isel(x=0, y=0)
    nwp = nwp.isel(init_time=list(range(0, 20)))

    # nwp = nwp.rename(name_dict={"UKV": id})

    # drop x and y
    nwp = nwp.drop_vars("x")
    nwp = nwp.drop_vars("y")
    nwp = nwp.assign_coords({"id": id})

    print("Loading NWP data")
    nwp.load()
    nwp = nwp.UKV

    print(nwp)
    # save to file
    filename = f"data/{id}.netcdf"
    print(f"Saving to {filename}")
    nwp.to_netcdf(filename, engine="h5netcdf")


# loop over all sites
for i in tqdm(range(len(locations))):

    location = locations.iloc[i]
    nwp = nwp_all
    x_osgb = location.x_osgb
    y_osgb = location.y_osgb
    id = int(location.ss_id)

    print(id)
    print(f"Doing {i} out of {len(locations)}")

    filename = f"data/{id}.netcdf"
    if os.path.exists(filename):
        print(f"File ({filename}) already exists")
    else:
        extract_one_location(nwp, x_osgb, y_osgb, id)

# One site this is about 1MB for 2020 and 2021
