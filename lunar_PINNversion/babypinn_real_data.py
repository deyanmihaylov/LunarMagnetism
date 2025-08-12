import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as pl
from lunar_PINNversion.model import PINN
from lunar_PINNversion.dataloader.dataLoader import Lunar_data_loader
from lunar_PINNversion.dataloader.util import spherical_to_cartesian


R_lunar = 1701e3 # lunar radius
height_obs = 1e5 # height of observation

data_filename = '/home/memolnar/Projects/lunarmagnetism/data/Moon_Mag_100km.txt'

Lunar_data_loader1 = Lunar_data_loader(filename=data_filename)
# Random points inside the domain [0, 1]^3

domain = np.random.rand(2000000, 3) # Random points inside the domain [0, 1]^3

domain[:, 0] = domain[:, 0] * 1e5 + R_lunar # r in kkm
domain[:, 1] = domain[:, 1] * np.pi  - np.pi/2# theta
domain[:, 2] = domain[:, 2] * 2 * np.pi  - np.pi# theta

domain_xyz = np.array([spherical_to_cartesian(el[0] / (R_lunar), el[1], el[2]) for
                       el in domain])
domain_xyz = torch.tensor(domain_xyz, dtype=torch.float32).to(device)

boundary_points = np.stack((Lunar_data_loader1.x_coord,
                            Lunar_data_loader1.y_coord,
                            Lunar_data_loader1.z_coord), axis=-1) / (R_lunar)

boundary_points = torch.tensor(boundary_points, dtype=torch.float32).to(device)

B_measured = np.stack((Lunar_data_loader1.b_x,
                       Lunar_data_loader1.b_y,
                       Lunar_data_loader1.b_z), axis=-1)
B_measured = torch.tensor(B_measured, dtype=torch.float32).to(device)

inner_dataset = TensorDataset(domain_xyz)
inner_loader = DataLoader(inner_dataset, batch_size=24024,shuffle=True)

boundary_dataset = TensorDataset(boundary_points, B_measured)
boundary_loader = DataLoader(boundary_dataset, batch_size=24024, shuffle=True)

# Training script configuration
hidden_size = 256
num_freqs = 12 # Number of frequencies for positional encoding
max_freq = 4
input_size = 3  # for (x, y, z) coordinates
output_size = 1  # for the scalar magnetic potential

pinn = PINN(input_size, hidden_size, output_size, num_freqs, max_freq,
            device=device,
            use_siren=True, siren_hidden_layers=4, siren_w0=2.0)
pinn = pinn.to(device)

pinn.train_pinn(inner_loader, boundary_loader, Lunar_data_loader1,
                epochs=10000, lr=5e-4,
                lambda_bc=1.0, lambda_domain=1, period_eval=1,
                step_size=1, gamma=0.8)
