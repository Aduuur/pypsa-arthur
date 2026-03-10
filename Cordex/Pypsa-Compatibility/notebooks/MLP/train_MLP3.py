import xarray as xr
import numpy as np
import dask.array as da
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer
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

# --------- Lag-Features (7 Tage + 3-Tage-Differenz) ---------
n_lags = 30
X_lagged = []
y_lagged = y_full[n_lags:]  # Ziel ab n_lags
for i in range(n_lags, num_times):
    lags = []
    for j in range(n_lags):
        lags.append(X_full[i-j-1])
    diff = X_full[i-1] - X_full[i-4]  # 3-Tage-Differenz
    X_lagged.append(np.concatenate(lags + [diff]))
X_lagged = np.array(X_lagged)

# --------- Train/Test Split ---------
split = int(0.8 * X_lagged.shape[0])
X_train, X_test = X_lagged[:split], X_lagged[split:]
y_train, y_test = y_lagged[:split], y_lagged[split:]

# --------- Standardisierung / Quantile für robustere Skalierung ---------
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

batch_size = 64
train_loader = DataLoader(LagDataset(X_train_scaled, y_train_scaled),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(LagDataset(X_test_scaled, y_test_scaled),
                         batch_size=batch_size, shuffle=False)

# --------- Optimierte MLP Architektur ---------
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

# --------- Training ---------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_epochs = 100
patience = 10
best_loss = np.inf
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
    
    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
    val_loss /= len(test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    scheduler.step(val_loss)
    
    # EarlyStopping
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_model_best_MLP3.pt')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

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

# Smoothe Vorhersage (rolling 3 Tage)
pred_test_smooth = pd.DataFrame(pred_test).rolling(3, axis=0, min_periods=1).mean().values

test_mse = np.mean((pred_test_smooth - y_test)**2)
test_mae = np.mean(np.abs(pred_test_smooth - y_test))
test_corr = np.corrcoef(pred_test_smooth.flatten(), y_test.flatten())[0,1]

print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Correlation: {test_corr:.4f}")

# --------- Modell & Scaler speichern ---------
model_path = '/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/soiltemp_model_MLP3.pt'
torch.save(model.state_dict(), model_path)
with open('/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/scaler_X_MLP3.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('/mnt/speicher/.wissmit/paul/cd2es/notebooks/MLP/scaler_y_MLP3.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("Best model and scalers saved!")
