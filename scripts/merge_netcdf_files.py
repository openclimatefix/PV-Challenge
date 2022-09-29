import glob
import xarray as xr
import click

default_folder = "../data"


@click.command()
@click.option(
    "--folder",
    default=default_folder,
    type=click.STRING,
    help="The folder to look in, to change the parquet files.",
)
@click.option(
    "--filename",
    default="all.netcdf",
    type=click.STRING,
    help="The output filename (should end in .parquet)",
)
def main(folder: str, filename: str):

    # get all file names
    files = glob.glob(folder + "/*.netcdf")
    files = sorted(files)

    print(f"Found {len(files)} files in {folder}")

    data_all_xr = []
    for file in files:
        print(f"loading {file}")
        data_xr = xr.open_dataset(file)
        data_all_xr.append(data_xr)

    print("merging file")
    data_all_xr = xr.concat(data_all_xr, dim='id')
    print(data_all_xr)

    print(f"Saving")
    data_all_xr.to_netcdf(filename, engine="h5netcdf")


if __name__ == "__main__":
    main()
