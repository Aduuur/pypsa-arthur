# cd2es - covert cordex climate data to energy system input data

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Warning: Experimental functionality. Use with caution!

Predicts (since not available in CORDEX) soil temperature (tso) from air temperature (tas) data using a pre-trained neural network model.

The script performs the following steps:
1. Loads daily air temperature data and aggregates it to daily means.
2. Loads a pre-trained neural network model and the corresponding scalers for input and output data.
3. Uses the model to predict daily soil temperature.
4. Smooths the predicted soil temperature data over a specified number of days.
5. Interpolates the daily soil temperature predictions to hourly data.
6. Saves the hourly soil temperature predictions to a NetCDF file.

"""

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd


airtemp_predictor =  snakemake.input.climvar_tas
output_file = snakemake.output[0]
checkpoint_path = snakemake.input.MLP_tso
smooth_days = 10 #smooth predicted time series


ds_air = xr.open_dataset(airtemp_predictor)
ds_air = ds_air.chunk({"time": 876})
air_daily = ds_air['tas'].resample(time='1D').mean(dim='time') #aggregate daily

num_times = air_daily.sizes['time']
num_lat = air_daily.sizes['lat']
num_lon = air_daily.sizes['lon']

X = air_daily.values.reshape(num_times, num_lat*num_lon)

# --------- Load model and scaler ---------

checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

X_scaled = scaler_X.transform(X)

class SoilTempMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_scaled.shape[1]
output_dim = X_scaled.shape[1]  # same dimension as input

model = SoilTempMLP(input_dim, output_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --------- Predict ---------
with torch.no_grad():
    X_tensor = torch.from_numpy(X_scaled).float()
    y_pred_scaled = model(X_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_pred_daily = y_pred.reshape(num_times, num_lat, num_lon)


# --------- Safe as NetCDF (Daily) ---------
ds_pred = xr.Dataset(
    {"tso": (("time", "lat", "lon"), y_pred_daily)},
    coords={
        "time": air_daily['time'],
        "lat": air_daily['lat'],
        "lon": air_daily['lon']
    }
)#

# ------- smoothing -----
ds_pred_smooth = ds_pred['tso'].rolling(time=smooth_days, center=True, min_periods=1).mean()

# --------- Interpolate daily to hourly ---------
hourly_time = pd.date_range(start=ds_pred_smooth.time[0].values,
                            end=ds_pred_smooth.time[-1].values,
                            freq='1h')

# Interpolate
ds_pred_hourly = ds_pred_smooth.interp(time=hourly_time)

# --------- Save hourly predictions as NetCDF ---------
ds_pred_hourly.to_netcdf(output_file)
print(f"Predicted soil temperature interpolated to hourly saved under: {output_file}")

