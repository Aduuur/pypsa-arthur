import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

# --------- Datei laden ---------
air_file_2015 = "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2015.nc"
output_file = "/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_2015_pred.nc"

ds_air = xr.open_dataset(air_file_2015, chunks={"valid_time": 876})
air_daily = ds_air['t2m'].resample(valid_time='1D').mean(dim='valid_time')

num_times = air_daily.sizes['valid_time']
num_lat = air_daily.sizes['latitude']
num_lon = air_daily.sizes['longitude']

X = air_daily.values.reshape(num_times, num_lat*num_lon)

# --------- Modell und Scaler laden ---------
checkpoint_path = "/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_model_full_with_scalers.pt"
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
output_dim = X_scaled.shape[1]  # gleiche Dimension wie Input

model = SoilTempMLP(input_dim, output_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --------- Vorhersage ---------
with torch.no_grad():
    X_tensor = torch.from_numpy(X_scaled).float()
    y_pred_scaled = model(X_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_pred_daily = y_pred.reshape(num_times, num_lat, num_lon)

# --------- Als NetCDF speichern ---------
ds_pred = xr.Dataset(
    {"soiltemp": (("valid_time", "latitude", "longitude"), y_pred_daily)},
    coords={
        "valid_time": air_daily['valid_time'],
        "latitude": air_daily['latitude'],
        "longitude": air_daily['longitude']
    }
)

ds_pred.to_netcdf(output_file)
print(f"Vorhersage gespeichert unter: {output_file}")
