"""
Switch to pytorch lightning model

Try using NWP and PV history to predict PV future

"""

import logging

import pandas as pd
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ocf_datapipes.training.nwp_pv import nwp_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey
from plotly.subplots import make_subplots
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

default_output_variable = "pv_yield"
import numpy as np
np.random.seed(2)

# set up logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)


# make iterattor
nwp_pv = nwp_pv_datapipe("experiments/002/exp_002.yaml")

dl = DataLoader(dataset=nwp_pv, batch_size=None)
nwp_pv = iter(dl)

# get a batch
batch = next(nwp_pv)


def plot(batch, y_hat):
    y = batch[BatchKey.pv][:, :, 0]
    time_y_hat = batch[BatchKey.pv_time_utc][:, batch[BatchKey.pv_t0_idx] :]
    time = batch[BatchKey.pv_time_utc]
    ids = batch[BatchKey.pv_id].detach().numpy()
    ids = [str(id) for id in ids]

    fig = make_subplots(rows=4, cols=4,subplot_titles=ids)

    for i in range(0, 16):
        row = i % 4 + 1
        col = i // 4 + 1
        time_i = pd.to_datetime(time[i], unit="s")
        time_y_hat_i = pd.to_datetime(time_y_hat[i], unit="s")
        trace_1 = go.Scatter(
            x=time_i, y=y[i].detach().numpy(), name="truth", line=dict(color="blue")
        )
        trace_2 = go.Scatter(
            x=time_y_hat_i, y=y_hat[i].detach().numpy(), name="predict", line=dict(color="red")
        )

        fig.add_trace(trace_1, row=row, col=col)
        fig.add_trace(trace_2, row=row, col=col)

    fig.update_yaxes(range=[0, 1])

    fig.show(renderer="browser")


def batch_to_x(batch):

    pv_t0_idx = batch[BatchKey.pv_t0_idx]
    nwp_t0_idx = batch[BatchKey.nwp_t0_idx]

    # x,y locations
    x_osgb = batch[BatchKey.pv_x_osgb] / 10**6
    y_osgb = batch[BatchKey.pv_y_osgb] / 10**6

    # add pv capacity
    pv_capacity = batch[BatchKey.pv_capacity_watt_power]/1000

    # future sun
    sun = batch[BatchKey.pv_solar_elevation][:, pv_t0_idx:]
    sun_az = batch[BatchKey.pv_solar_azimuth][:, pv_t0_idx:]

    # future nwp
    nwp = batch[BatchKey.nwp][:, nwp_t0_idx:]
    nwp = nwp.reshape([nwp.shape[0], nwp.shape[1] * nwp.shape[2]])

    # history pv
    pv = batch[BatchKey.pv][:, :pv_t0_idx, 0]
    x = torch.concat((pv, nwp, sun, sun_az, x_osgb, y_osgb, pv_capacity), dim=1)

    return x


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def _training_or_validation_step(self, x, return_model_outputs: bool = False):
        """
        batch: The batch data
        tag: either 'Train', 'Validation' , 'Test'
        """

        # put the batch data through the model
        y_hat = self(x)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = x[BatchKey.pv][:, x[BatchKey.pv_t0_idx] :, 0]

        # calculate mse, mae
        mse_loss = torch.nn.MSELoss()(y_hat, y)
        # TODO, why does this work better? SOmething to do with sigmoid
        mae_loss = (y_hat - y).abs().mean()
        bce_loss = torch.nn.BCELoss()(y_hat,y)

        loss = bce_loss

        self.log('mse', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('mae', mae_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('bce', bce_loss, on_step=True, on_epoch=True, prog_bar=True)

        if return_model_outputs:
            return loss, y_hat
        else:
            return loss

    def training_step(self, x, batch_idx):

        if batch_idx < 1:
            plot(x, self(x))

        return self._training_or_validation_step(x)

    def validation_step(self, x, batch_idx):

        if batch_idx < 1:
            plot(x, self(x))

        return self._training_or_validation_step(x)

    def predict_step(self, x, batch_idx, dataloader_idx=0):
        return x, self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class Model(BaseModel):
    def __init__(self, input_length, output_length):
        super().__init__()
        features= 256
        self.input_length = input_length
        self.fc1 = nn.Linear(in_features=self.input_length, out_features=features)
        self.fc2 = nn.Linear(in_features=features, out_features=features)
        self.fc3 = nn.Linear(in_features=features, out_features=features)

        self.pv_system_id_embedding = nn.Embedding(num_embeddings=10000, embedding_dim=16)
        self.fc4 = nn.Linear(in_features=features + 16, out_features=output_length)

    def forward(self, x):
        x = batch_to_x(x)

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        id_embedding = self.pv_system_id_embedding(batch[BatchKey.pv_id].type(torch.IntTensor))
        id_embedding = id_embedding.squeeze(1)
        out = torch.concat([out, id_embedding], dim=1)

        out = torch.sigmoid(self.fc4(out))
        # out = self.fc4(out)

        return out


# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    devices=None,
    max_epochs=3,
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_batches: int):

        self.n_batches = n_batches

    def __len__(self):
        """Length of dataset"""
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> dict:

        batch = next(nwp_pv)

        return batch


train_loader = DataLoader(Dataset(n_batches=100), batch_size=None, num_workers=0)
val_loader = DataLoader(Dataset(n_batches=1), batch_size=None, num_workers=0)
predict_loader = DataLoader(Dataset(n_batches=1), batch_size=None, num_workers=0)

x = batch_to_x(batch)
y = batch[BatchKey.pv][:, batch[BatchKey.pv_t0_idx] :, 0]
input_length = x.shape[1]
output_length = y.shape[1]

model = Model(input_length=input_length, output_length=output_length)


trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# predict model for some plots
batch = next(nwp_pv)
y_hat = model(batch)

plot(batch, y_hat)

##### plot results
plot(batch, y_hat)

# ***** Brief results ******
# just on training data 3 epochs (max 40 batches, batch_size=4)

# pv history, sun and nwp (4 hour)
# loss = 0.264

# pv history, sun and nwp (24 hours)
# loss = 0.277

# pv history (48 hours), sun and nwp (forecast 24 hours)
# loss = 0.277


# begun to notice some odd data
# Idea: drop data if 24 hours of zeros
# pv history (48 hours), sun and nwp (forecast 24 hours)

# also solved bug in ocf peipes with yield
# loss = 0.246

# Batch size 32
# loss = 0.244
