import xarray as xr
import numpy as np
import torch
import pickle

# --- Parameters ---
n_lags = 3
air_2015_file = "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2015.nc"

scaler_X_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/scaler_X_MLP2.pkl'
scaler_y_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/scaler_y_MLP2.pkl'
model_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/soiltemp_model_MLP2.pt'
output_nc = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/soiltemp_pred_2015_MLP2.nc'

# --- Load air temp and aggregate ---
air_2015 = xr.open_dataset(air_2015_file)
air_2015_daily = air_2015['t2m'].resample(valid_time='1D').mean(dim='valid_time')
num_times = air_2015_daily.sizes['valid_time']
num_lat = air_2015_daily.sizes['latitude']
num_lon = air_2015_daily.sizes['longitude']
X_2015_full = air_2015_daily.values.reshape(num_times, num_lat*num_lon)

# --- Lag features ---
X_2015_lagged = []
for i in range(n_lags, num_times):
    lags = []
    for j in range(n_lags):
        lags.append(X_2015_full[i-j-1])
    X_2015_lagged.append(np.concatenate(lags))
X_2015_lagged = np.array(X_2015_lagged)

# --- Load scalers and model ---
with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)
X_2015_lagged_scaled = scaler_X.transform(X_2015_lagged)

input_dim = X_2015_lagged_scaled.shape[1]
output_dim = scaler_y.mean_.shape[0]

class SoilTempMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SoilTempMLP(input_dim, output_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Predict ---
with torch.no_grad():
    X_tensor = torch.from_numpy(X_2015_lagged_scaled).float().to(device)
    pred_scaled = model(X_tensor).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_scaled)

# --- Reshape and save ---
num_pred_days = pred.shape[0]
pred_reshaped = pred.reshape(num_pred_days, num_lat, num_lon)
time_pred = air_2015_daily['valid_time'].values[n_lags:]

soil_pred_da = xr.DataArray(
    pred_reshaped,
    dims=['valid_time', 'latitude', 'longitude'],
    coords={
        'valid_time': time_pred,
        'latitude': air_2015_daily['latitude'].values,
        'longitude': air_2015_daily['longitude'].values
    },
    name='soiltemp_pred'
)

soil_pred_da.to_netcdf(output_nc)
print(f"Saved predictions to {output_nc}")