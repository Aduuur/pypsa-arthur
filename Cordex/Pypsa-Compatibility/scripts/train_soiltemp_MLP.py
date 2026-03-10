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

Script for training a Multilayer Perceptron (MLP) model to predict soil temperature from air temperature.
Air temperature and soil temperature data from ERA5 are used to train and verify the model. 
This is utilized in build_soiltep.py to estimate soil temperature from CORDEX air temperature, as the latter does not contain soil temperature data.

This script performs the following steps:
1. Aggregates hourly climate data to daily data.
2. Prepares and scales the data for training and testing.
3. Defines and trains an MLP model using the training data.
4. Evaluates the model using the test data.
5. Plots and saves the prediction results.
"""

import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
import math
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


soil_files = snakemake.input.climvar_tso_hist
air_files = snakemake.input.climvar_tas_hist

batch_size = 64
num_workers = math.ceil(snakemake.threads/3) 
num_threads_torch=snakemake.threads
test_year = int(soil_files[-1][-7:-3]) #last year of files is used as test
smooth_days = 10 #smooth predicted time series

MLP_model_path=snakemake.output.MLP_model
MLP_training_log=snakemake.output.MLP_model_log
MLP_predict_plot_file = snakemake.output.MLP_predict_plot_file


# --------- Set up logging ---------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MLP_training_log),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

logger.info(f'Used: \n Batch size: {batch_size} \n Number of workers: {num_workers} \n Number of threds for torch: {num_threads_torch} \n Test year: {test_year}')


# --------- Aggregation from hourly to daily ---------
def aggregate_daily(file_list, var_name):
    daily_data = []
    for f in file_list:
        logger.info(f'Open file: {f}')
        ds = xr.open_dataset(f)
        ds = ds.chunk({"valid_time": 876})
        daily = ds[var_name].resample(valid_time='1D').mean(dim='valid_time')
        daily_data.append(daily)
    return xr.concat(daily_data, dim='valid_time')

soil_daily = aggregate_daily(soil_files, 'stl4')
air_daily = aggregate_daily(air_files, 't2m')

# --------- Data preperation ---------
num_times = soil_daily.sizes['valid_time']
num_lat = soil_daily.sizes['latitude']
num_lon = soil_daily.sizes['longitude']

X_full = air_daily.values.reshape(num_times, num_lat*num_lon)
y_full = soil_daily.values.reshape(num_times, num_lat*num_lon)

# --------- Use one year as test year ---------
dates = pd.to_datetime(soil_daily['valid_time'].values)
test_idx = dates.year == test_year

X_train = X_full[~test_idx]
y_train = y_full[~test_idx]

X_test = X_full[test_idx]
y_test = y_full[test_idx]

# --------- Scale ---------
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


train_loader = DataLoader(FullTimeDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(FullTimeDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# --------- MLP model ---------
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
torch.set_num_threads(num_threads_torch)

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
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        # Save the best model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }, MLP_model_path)
    else:
        counter += 1
        if counter >= patience:
            logger.info("Early stopping triggered")
            break

logger.info(f"Training finished in {time.time() - start_time:.2f} sec")

# --------- Load the best model with scalers ---------
checkpoint = torch.load(MLP_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# --------- Test evaluation ---------
logger.info(f'Test model with data from {test_year}')
model.eval()
pred_list = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        pred_list.append(pred)

pred_test_scaled = np.vstack(pred_list)
pred_test = scaler_y.inverse_transform(pred_test_scaled)
pred_test_smooth = pd.DataFrame(pred_test).rolling(smooth_days, axis=0, min_periods=1).mean().values

test_mse = np.mean((pred_test_smooth - y_test)**2)
test_mae = np.mean(np.abs(pred_test_smooth - y_test))
test_corr = np.corrcoef(pred_test_smooth.flatten(), y_test.flatten())[0,1]


# --------- Plot predictions vs. true values ---------
middle=int(len(pred_test_smooth[0,:])/2) #select grid point in middle examplary
pred_test_smooth=list(pred_test_smooth[:,middle])
y_test=list(y_test[:,middle])

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="True", color="blue")
plt.plot(pred_test_smooth, label="Predicted", color="red", alpha=0.7)
plt.xlabel("Time step")
plt.ylabel("Soil Temperature")
plt.title(f"Soil Temperature Prediction for example point in middle for {test_year}\nMSE: {test_mse:.4f}, MAE: {test_mae:.4f}, Corr: {test_corr:.4f}")
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(MLP_predict_plot_file, format='pdf')
plt.close()
logger.info(f"Prediction plot saved as PDF under: {MLP_predict_plot_file}")


# --------- Log infos ---------
logger.info(f"Test MSE: {test_mse:.4f}")
logger.info(f"Test MAE: {test_mae:.4f}")
logger.info(f"Test Correlation: {test_corr:.4f}")
logger.info('Training finished!')