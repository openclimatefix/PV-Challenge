# PV-Challenge

At OCF we are building a state of the art national solar generation forecast that will be used by National Grid.
By improving the solar generation forecast, we can help them to minimise the use of standby gas turbines,
potentially leading to a substantial reduction in carbon emissions.

## Challenge

Build the best PV forecasting model you can.
The model should forecast a PV system for the next 24 hours. The forecast should be in 30 minutes intervals, and show the average power produced in that 30 minute interval. 

We would recommend that you use current data from other PV systems to help improve the forecast.

## Data

A PV dataset is available [here](https://huggingface.co/datasets/openclimatefix/uk_pv)
or [here](https://console.cloud.google.com/marketplace/product/bigquery-public-data/eumetsat-seviri-rss-hrv-uk).
This dataset contains data from 1311 PV systems from 2018-01-01 to 2021-10-27. More data will be available soon.

We would recommend splitting the data into 2018 for training and 2019 for testing.
Validation splits can be made how you wish in the training subset.

You can of course use other data, for example
[satellite data](https://huggingface.co/datasets/openclimatefix/eumetsat_uk_hrv), or NWP data,
but you are welcome to use other sources

### Metrics

For now lets use MAE, but perhaps we can explore a few other metrics that help differentiate
a excellent model, from a good model

# How

Please contact [OCF](info@openclimatefix.org) if you like to contribute, or make a PR.
We are interested in getting lots of people involved and building a great solution together.


## Files

### Data

The data folder contains the locations of the pv sites from [the dataset](https://huggingface.co/datasets/openclimatefix/uk_pv)

### Scripts

There is a script to extract NWP data from a large file. This file unfortunately is not public. 
