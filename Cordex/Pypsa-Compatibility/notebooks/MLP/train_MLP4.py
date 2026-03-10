import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import time
import pandas as pd

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

soil_daily = aggregate_daily(soil_files, 'stl4')
air_daily = aggregate_daily(air_files, 't2m')

# --------- Daten vorbereiten ---------
num_times = soil_daily.sizes['valid_time']
num_lat = soil_daily.sizes['latitude']
num_lon = soil_daily.sizes['longitude']

X_full = air_daily.values.reshape(num_times, num_lat*num_lon)
y_full = soil_daily.values.reshape(num_times, num_lat*num_lon)

# --------- Jahr als Test abtrennen ---------
dates = pd.to_datetime(soil_daily['valid_time'].values)
test_year = 2015
test_idx = dates.year == test_year

X_train = X_full[~test_idx]
y_train = y_full[~test_idx]

X_test = X_full[test_idx]
y_test = y_full[test_idx]

# --------- Skalierung ---------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# --------- Dataset ---------
class FullTimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
num_workers = 16
train_loader = DataLoader(FullTimeDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(FullTimeDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# --------- MLP-Modell ---------
class SoilTempMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]

model = SoilTempMLP(input_dim, output_dim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
torch.set_num_threads(48)

# --------- Training ---------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_epochs = 100
best_loss = np.inf
patience = 10
counter = 0

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
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_model_best_full.pt')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

print(f"Training finished in {time.time() - start_time:.2f} sec")

# --------- Testauswertung ---------
model.eval()
pred_list = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        pred_list.append(pred)

pred_test_scaled = np.vstack(pred_list)
pred_test = scaler_y.inverse_transform(pred_test_scaled)
pred_test_smooth = pd.DataFrame(pred_test).rolling(3, axis=0, min_periods=1).mean().values

test_mse = np.mean((pred_test_smooth - y_test)**2)
test_mae = np.mean(np.abs(pred_test_smooth - y_test))
test_corr = np.corrcoef(pred_test_smooth.flatten(), y_test.flatten())[0,1]

print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Correlation: {test_corr:.4f}")

# --------- Modell & Scaler speichern ---------
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y
}, '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_model_full_with_scalers.pt')

print("Best model and scalers saved!")
