"""
Try using NWP and PV history to print PV future

Just trying a really simple Keras model, to check flow of data works.

"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ocf_datapipes.training.nwp_pv import nwp_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import Dense
import logging

import numpy as np
import pandas as pd

# set up logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)


# make iterattor
simple_pv = nwp_pv_datapipe("experiments/001/exp_001.yaml")
simple_pv = iter(simple_pv)

# get a batch
batch = next(simple_pv)


class DataIterator(Sequence):
    def __init__(self, max=10):
        self.max = max

    def __iter__(self):
        return self

    def __getitem__(self, index):
        batch = next(simple_pv)

        # pv dataset
        # X = batch[BatchKey.pv][:,:batch[BatchKey.pv_t0_idx]:][:,:,0]

        # use sun elevation
        # X = batch[BatchKey.pv_solar_elevation][:, batch[BatchKey.pv_t0_idx]:]

        # use sun and nwp elevation
        # sun = batch[BatchKey.pv_solar_elevation][:, batch[BatchKey.pv_t0_idx] :]
        # nwp = batch[BatchKey.nwp][:, batch[BatchKey.nwp_t0_idx] :]
        # nwp = nwp.reshape([nwp.shape[0], nwp.shape[1] * nwp.shape[2]])
        # X = np.concatenate((sun, nwp), axis=1)

        # use pv, sun and nwp elevation
        sun = batch[BatchKey.pv_solar_elevation][:, batch[BatchKey.pv_t0_idx] :]
        nwp = batch[BatchKey.nwp][:, batch[BatchKey.nwp_t0_idx] :]
        nwp = nwp.reshape([nwp.shape[0], nwp.shape[1] * nwp.shape[2]])
        pv = batch[BatchKey.pv][:, :batch[BatchKey.pv_t0_idx]:][:, :, 0]
        X = np.concatenate((sun, nwp, pv), axis=1)

        # true pv value
        y = batch[BatchKey.pv][:, batch[BatchKey.pv_t0_idx] :][:, :, 0]
        return X, y

    def on_epoch_end(self):
        pass

    def __len__(self):
        return self.max
        # else:
        #     raise StopIteration


d = DataIterator(max=40)
x, y = d.__getitem__(1)

input_length = x.shape[1]

model = Sequential()
model.add(Dense(64, input_shape=(input_length,), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(289, activation="sigmoid"))
# compile the keras model

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
# fit the keras model on the dataset
model.fit(d, epochs=5, batch_size=None)

y_hat = model.predict(x)

##### plot results

fig = make_subplots(rows=2, cols=2)

for i in range(0, 4):

    row = i % 2 + 1
    col = i // 2 + 1
    trace_1 = go.Scatter(y=y[i], name="truth", line=dict(color="blue"))
    trace_2 = go.Scatter(y=y_hat[i], name="predict", line=dict(color="red"))

    fig.add_trace(trace_1, row=row, col=col)
    fig.add_trace(trace_2, row=row, col=col)

fig.update_yaxes(range=[0, 1])

fig.show(renderer="browser")

# ***** Brief results ******
# just on training data 5 epochs (max 40 batches, batch_size=4)

# pv history
# loss = 0.0303, 0.0254, 0.0221, 0.0198, 0.0191

# sun
# loss = 0.0738, 0.0367, 0.0300, 0.0291, 0.0224

# sun and nwp
# loss = 0.0555, 0.0420, 0.0349, 0.0283, 0.261

# pv history, sun and nwp
# loss = 0.0452, 0.0344, 0.0268, 0.0190, 0.0219
# with sigmoid loss function
# loss = 0.0904, 0.0302, 0.0258, 0.0169, 0.0178

