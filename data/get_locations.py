# get a list of sites that we have 5 mins data
# Will be using data from here - https://huggingface.co/datasets/openclimatefix/uk_pv
# We need to download the pv.netcdf from HF and put it in the data folder

import pandas as pd
import xarray as xr
import ssl
from ocf_datapipes.utils.geospatial import lat_lon_to_osgb

# make sure SLL doesnt stop us reading the data
ssl._create_default_https_context = ssl._create_unverified_context

# read meta file
metadata = pd.read_csv("https://huggingface.co/datasets/openclimatefix/uk_pv/raw/main/metadata.csv")

# load the pv data
pv_power = xr.open_dataset("data/pv.netcdf", engine="h5netcdf")
site_ids = list(pv_power.keys())

# make sure site ids are integers
site_ids = [int(site_id) for site_id in site_ids]

# select the metadata that we have data fore
metadata = metadata[metadata["ss_id"].isin(site_ids)]

# get osgb coorindates
metadata["x_osgb"], metadata["y_osgb"] = lat_lon_to_osgb(
    latitude=metadata["latitude_rounded"], longitude=metadata["longitude_rounded"]
)

# save details we need
metadata = metadata[["ss_id", "x_osgb", "y_osgb"]]
metadata.to_csv("data/locations.csv", index=False)
