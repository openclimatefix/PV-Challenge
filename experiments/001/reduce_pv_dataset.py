""" File to reduce PV data to the sites in the nwp file"""

import xarray as xr
import pandas as pd

nwp = xr.open_dataset(
    "experiments/001/nwp.netcdf",
    engine="h5netcdf",
    chunks="auto",
)

ids = nwp.id.values

metadata = pd.read_csv("data/metadata_passiv.csv")
metadata = metadata[metadata["ss_id"].isin(ids)]
metadata.to_csv("experiments/001/metadata_passiv.csv", index=False)

pv = xr.open_dataset(
    "data/pv.netcdf",
    engine="h5netcdf",
    chunks="auto",
)

ids = [str(id) for id in ids]
pv = pv[ids]
pv.to_netcdf('experiments/001/pv.netcdf',engine='h5netcdf')
