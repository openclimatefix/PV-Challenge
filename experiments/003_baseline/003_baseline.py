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

# set up logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)


# make iterattor
nwp_pv = nwp_pv_datapipe("experiments/003_baseline/exp_003.yaml")

dl = DataLoader(dataset=nwp_pv, batch_size=None)
nwp_pv = iter(dl)

# get a batch
# batch = next(nwp_pv)

class BaseModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def _training_or_validation_step(self, batch, return_model_outputs: bool = False):
        """
        batch: The batch data
        tag: either 'Train', 'Validation' , 'Test'
        """

        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch[BatchKey.pv][:, batch[BatchKey.pv_t0_idx]:, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = ((y_hat - y)**2).abs().mean()
        bce_loss = torch.nn.BCELoss()(y_hat, y)
        loss = bce_loss

        self.log('mae',loss, on_step=True, on_epoch=True,prog_bar=True)

        if return_model_outputs:
            return loss, y_hat
        else:
            return loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return batch, self(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        mae_loss = self._training_or_validation_step(batch)

        return mae_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


class Model(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        # return 0's
        out = torch.zeros_like(x[BatchKey.pv][:, :x[BatchKey.pv_t0_idx], 0]) + 0.01

        # return yesterday
        # out = x[BatchKey.pv][:, :x[BatchKey.pv_t0_idx], 0]
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


# datasets
train_loader = DataLoader(Dataset(n_batches=100), batch_size=None, num_workers=0)
predict_loader = DataLoader(Dataset(n_batches=1), batch_size=None, num_workers=0)

# get model
model = Model()

# test model
trainer.test(model, train_loader)

# predict model for some plots
batch = next(nwp_pv)
y_hat = model(batch)
y = batch[BatchKey.pv][:, :, 0]
time_y_hat = batch[BatchKey.pv_time_utc][:, batch[BatchKey.pv_t0_idx]:]
time = batch[BatchKey.pv_time_utc]

##### plot results
fig = make_subplots(rows=2, cols=2)

for i in range(0, 4):

    row = i % 2 + 1
    col = i // 2 + 1
    time_i = pd.to_datetime(time[i], unit="s")
    time_y_hat_i = pd.to_datetime(time_y_hat[i], unit="s")
    trace_1 = go.Scatter(x=time_i, y=y[i].detach().numpy(), name="truth", line=dict(color="blue"))
    trace_2 = go.Scatter(x=time_y_hat_i, y=y_hat[i].detach().numpy(), name="predict", line=dict(color="red"))

    fig.add_trace(trace_1, row=row, col=col)
    fig.add_trace(trace_2, row=row, col=col)

fig.update_yaxes(range=[0, 1])

fig.show(renderer="browser")

# ***** Brief results ******
# Tested with 400 samples
# project forward yesterday results gives a
# mse loss = 0.019, 0.021, 0.018   ( 3 runs)
# bcd loss = 0.272

# just returned zeros
# mse loss = 0.041, 0.045, 0.045   ( 3 runs)
# bcd loss = 0.485
