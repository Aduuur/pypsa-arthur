import xarray as xr
import numpy as np
import dask.array as da
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import time

# --------- Dateipfade ---------
soil_files = [
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/stl4_era5_2011.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/stl4_era5_2012.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/stl4_era5_2013.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/stl4_era5_2014.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/stl4_era5_2015.nc"
]
air_files = [
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2011.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2012.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2013.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2014.nc",
    "/mnt/speicher/.wissmit/paul/cdata_pypsa_grid/europe/observed/t2m_era5_2015.nc"
]

# --------- Aggregation stündlich -> täglich ---------
def aggregate_daily(file_list, var_name):
    daily_data = []
    for f in file_list:
        print(f'Öffne Datei: {f}', flush=True)
        ds = xr.open_dataset(f, chunks={"valid_time": 876})
        daily = ds[var_name].resample(valid_time='1D').mean(dim='valid_time')
        daily_data.append(daily)
    return xr.concat(daily_data, dim='valid_time')

print("Aggregating soil data...")
soil_daily = aggregate_daily(soil_files, 'stl4')
print("Aggregating air data...")
air_daily = aggregate_daily(air_files, 't2m')

# --------- Daten vorbereiten ---------
num_times = soil_daily.sizes['valid_time']
num_lat = soil_daily.sizes['latitude']
num_lon = soil_daily.sizes['longitude']

X_full = air_daily.values.reshape(num_times, num_lat*num_lon)
y_full = soil_daily.values.reshape(num_times, num_lat*num_lon)

# --------- Lag-Features ---------
n_lags = 3
X_lagged = []
y_lagged = y_full[n_lags:]  # Ziel ab n_lags
for i in range(n_lags, num_times):
    lags = []
    for j in range(n_lags):
        lags.append(X_full[i-j-1])
    X_lagged.append(np.concatenate(lags))
X_lagged = np.array(X_lagged)

# Train/Test Split
split = int(0.8 * X_lagged.shape[0])
X_train, X_test = X_lagged[:split], X_lagged[split:]
y_train, y_test = y_lagged[:split], y_lagged[split:]

# Standardisierung
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# --------- PyTorch Dataset ---------
class LagDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_loader = DataLoader(LagDataset(X_train_scaled, y_train_scaled),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(LagDataset(X_test_scaled, y_test_scaled),
                         batch_size=batch_size, shuffle=False)

# --------- MLP Netzwerk ---------
class SoilTempMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]

model = SoilTempMLP(input_dim, output_dim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# --------- Training ---------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

total_time = time.time() - start_time
print(f"Training finished in {total_time:.2f} sec")

# --------- Evaluation ---------
model.eval()
pred_list = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        pred_list.append(pred)
pred_test_scaled = np.vstack(pred_list)
pred_test = scaler_y.inverse_transform(pred_test_scaled)

test_mse = np.mean((pred_test - y_test)**2)
test_mae = np.mean(np.abs(pred_test - y_test))
test_corr = np.corrcoef(pred_test.flatten(), y_test.flatten())[0,1]

print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Correlation: {test_corr:.4f}")

# --------- Modell & Scaler speichern ---------
model_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_model_MLP2.pt'
scaler_X_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/scaler_X_MLP2.pkl'
scaler_y_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/scaler_y_MLP2.pkl'

torch.save(model.state_dict(), model_path)
with open(scaler_X_path, 'wb') as f:
    pickle.dump(scaler_X, f)
with open(scaler_y_path, 'wb') as f:
    pickle.dump(scaler_y, f)

print("Model and scalers saved!")
