import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as pl
import torch
import numpy as np
import wandb
from lunar_PINNversion.dataloader.util import spherical_to_cartesian

R_lunar = 1701e3 # km
class PositionalEncoding(nn.Module):

    def __init__(self, num_freqs, d_input, max_freq=8):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.frequencies = nn.Parameter(frequencies[None, :, None], requires_grad=False)
        self.d_output = d_input * (num_freqs * 2)

    def forward(self, x):
        encoded = x[:, None, :] * torch.pi * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


# ---- SIREN layer + utilities ----
class SIRENLinear(nn.Module):
    """
    Linear layer with SIREN-style initialization and sine activation applied in forward.
    If is_first the initialization uses a different range per the SIREN paper.
    """
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # Initialization per SIREN paper:
        with torch.no_grad():
            if is_first:
                # first-layer init: uniform(-1/in, 1/in)
                bound = 1.0 / in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # subsequent layers: uniform(-sqrt(6/in)/w0, sqrt(6/in)/w0)
                bound = (np.sqrt(6.0 / in_features) / self.w0)
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        # apply linear then sin(w0 * linear(x))
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    """
    Multi-layer SIREN block. Final layer is linear readout (no sine) unless keep_sine_on_final=True.
    """
    def __init__(self, in_dim, hidden_dim, n_layers, w0=30.0, w0_initial=30.0, keep_sine_on_final=False):
        """
        in_dim: input dim
        hidden_dim: hidden layer width
        n_layers: number of SIREN layers (excluding final linear readout)
        w0: w0 for subsequent layers
        w0_initial: w0 for first layer (often set to 30.0)
        """
        super().__init__()
        layers = []
        # first SIREN layer (special init + w0_initial)
        layers.append(SIRENLinear(in_dim, hidden_dim, w0=w0_initial, is_first=True))
        # remaining SIREN layers
        for _ in range(max(0, n_layers - 1)):
            layers.append(SIRENLinear(hidden_dim, hidden_dim, w0=w0, is_first=False))

        self.siren = nn.Sequential(*layers)
        # final readout: either linear (no sine) or SIRENLinear if requested
        if keep_sine_on_final:
            self.final = SIRENLinear(hidden_dim, 1, w0=w0, is_first=False)
        else:
            self.final = nn.Linear(hidden_dim, 1)
            # init final readout similar to small uniform
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_dim) / w0
                self.final.weight.uniform_(-bound, bound)
                if self.final.bias is not None:
                    self.final.bias.zero_()

    def forward(self, x):
        h = self.siren(x)
        return self.final(h)

# Define the neural network model
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_freqs, max_freq=8,
                 device='cpu',
                 use_siren=False,
                 siren_hidden_layers=3,
                 siren_w0=30.0,
                 keep_sine_on_final=False
                 ):
        super(PINN, self).__init__()
        self.device = device
        self.positional_encoding = PositionalEncoding(num_freqs, input_size, max_freq)
        in_dim = self.positional_encoding.d_output

        self.use_siren = use_siren

        if use_siren:
            # Build SIREN block. The SIREN returns a scalar (output_size==1 expected).
            # If your output_size > 1, you'd need to adapt final layer dimension.
            if output_size != 1:
                # support vector outputs by making final layer map to output_size
                # we achieve that by replacing SIREN.final with appropriate linear
                self.siren = SIREN(in_dim, hidden_size, siren_hidden_layers,
                                   w0=siren_w0, w0_initial=siren_w0, keep_sine_on_final=False)
                # replace final readout to map to output_size
                last_linear = nn.Linear(hidden_size, output_size)
                with torch.no_grad():
                    bound = np.sqrt(6.0 / hidden_size) / siren_w0
                    last_linear.weight.uniform_(-bound, bound)
                    if last_linear.bias is not None:
                        last_linear.bias.zero_()
                # chain: siren.siren -> last_linear
                self.hidden = nn.Sequential(self.siren.siren, last_linear)
            else:
                # single-output SIREN (common case for scalar potential)
                self.siren = SIREN(in_dim, hidden_size, siren_hidden_layers,
                                   w0=siren_w0, w0_initial=siren_w0, keep_sine_on_final=keep_sine_on_final)
                # expose a consistent API: self.hidden(x) returns shape (N, output_size)
                self.hidden = self.siren
        else:
            # original tanh MLP
            self.hidden = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, output_size)
            )

        # move to device
        self.to(self.device)

    def forward(self, x):
        if not x.is_floating_point():
            x = x.float()
        x = x.to(self.device)
        x_encoded = self.positional_encoding(x)
        out = self.hidden(x_encoded)
        # if output is (N,1) make sure it's (N,1) shape consistently
        return out

    # Compute the Laplacian using automatic differentiation
    def compute_laplacian(self, inputs):
        phi = self(inputs.requires_grad_(True))
        grad_phi = torch.autograd.grad(outputs=phi, inputs=inputs, grad_outputs=torch.ones_like(phi),
                                       create_graph=True)[0]
        laplacian = sum(
            [torch.autograd.grad(outputs=grad_phi[:, i], inputs=inputs, grad_outputs=torch.ones_like(grad_phi[:, i]),
                                 create_graph=True)[0][:, i] for i in range(3)])
        return laplacian


    def boundary_condition_loss(self, inputs, B_measured):
        phi = self(inputs.requires_grad_(True))
        grad_phi = torch.autograd.grad(outputs=phi, inputs=inputs, grad_outputs=torch.ones_like(phi),
                                       create_graph=True)[0]
        B_pred = -1 * grad_phi

        return torch.sum(((B_measured - B_pred)/(torch.abs(B_pred)+1e-3)) ** 2)

    def train_pinn(self, inner_loader, boundary_loader, lunar_data, epochs, lr,
                   lambda_domain=1, lambda_bc=1, period_log=1000, period_eval=5000,
                   step_size=1000, gamma=0.95):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


        for epoch in range(epochs):
            total_loss_epoch = 0.0
            progress_bar = tqdm(total=len(inner_loader) + len(boundary_loader),
                                desc=f"Epoch {epoch + 1}/{epochs}",
                                unit="batch",
                                leave=False)  # leave=True if you want to keep history
            running_laplacian_loss = 0.0
            running_bc_loss = 0.0
            # --- Loop over boundary points ---
            for (x_boundary_batch, B_batch) in boundary_loader:
                optimizer.zero_grad()
                boundary_loss = lambda_bc * self.boundary_condition_loss(x_boundary_batch, B_batch)
                running_bc_loss = boundary_loss.item()
                boundary_loss.backward()
                optimizer.step()
                total_loss_epoch += boundary_loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "laplacian": f"{running_laplacian_loss:.3e}",
                    "bc": f"{running_bc_loss:.3e}",
                    "total": f"{total_loss_epoch:.3e}"
                })
            # --- Loop over inner points ---
            for (x_inner_batch,) in inner_loader:
                optimizer.zero_grad()
                laplacian_loss = lambda_domain * torch.mean(self.compute_laplacian(x_inner_batch) ** 2)
                running_laplacian_loss = laplacian_loss.item()
                laplacian_loss.backward()
                optimizer.step()
                total_loss_epoch += laplacian_loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "lc": f"{running_laplacian_loss:.3e}",
                    "bc": f"{running_bc_loss:.3e}",
                    "total": f"{total_loss_epoch:.3e}"
                })

            progress_bar.close()
            if epoch % period_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Total Loss: {total_loss_epoch:.3e}, LR: {current_lr:.3e}")

            if epoch % period_eval == 0:
                self.evaluate_model(epoch, lunar_data)

            scheduler.step()

    def plot_B_eval(self, epoch, lunar_data):
        num_pts = 100

        def spherical_to_cartesian(r, theta, phi):
            x = r * torch.cos(theta) * torch.cos(phi)
            y = r * torch.cos(theta) * torch.sin(phi)
            z = r * torch.sin(theta)
            return x, y, z

        theta_linspace = torch.linspace(-torch.pi/2, torch.pi/2,num_pts)
        phi_linspace = torch.linspace(-np.pi, torch.pi, num_pts)
        r_lunar_surface = torch.ones(1)
        r_BC_surface = torch.ones(1) + 1e5/R_lunar

        # Create meshgrid in spherical coordinates
        # meshgrid(..., indexing='ij') gives shape [r, theta, phi]
        R_lunar_surface, Theta, Phi = torch.meshgrid(r_lunar_surface, theta_linspace, phi_linspace,
                                                     indexing='ij')
        R_orbit_surface, Theta, Phi = torch.meshgrid(r_BC_surface, theta_linspace, phi_linspace,
                                                     indexing='ij')
        # Convert spherical to Cartesian (vectorized)
        X_0, Y_0, Z_0 = spherical_to_cartesian(R_lunar_surface, Theta, Phi)
        X_BC, Y_BC, Z_BC = spherical_to_cartesian(R_orbit_surface, Theta, Phi)
        # Stack into single tensor if needed
        grid_mesh_eval_xyz_0 = torch.stack((X_0.ravel(), Y_0.ravel(), Z_0.ravel()), dim=-1)
        grid_mesh_eval_xyz_0 = torch.tensor(grid_mesh_eval_xyz_0, dtype=torch.float32,
                               requires_grad=True).to(self.device)
        phi_pred_0 = self(grid_mesh_eval_xyz_0)
        grad_phi_0 = torch.autograd.grad(outputs=phi_pred_0, inputs=grid_mesh_eval_xyz_0,
                                       grad_outputs=torch.ones_like(phi_pred_0),
                                       create_graph=True)[0]

        B_pred_0 = (-1 * grad_phi_0).cpu().detach().numpy()

        grid_mesh_eval_xyz_BC = torch.stack((X_BC.ravel(), Y_BC.ravel(), Z_BC.ravel()), dim=-1)
        grid_mesh_eval_xyz_BC = torch.tensor(grid_mesh_eval_xyz_BC, dtype=torch.float32,
                                            requires_grad=True).to(self.device)
        phi_pred_BC = self(grid_mesh_eval_xyz_BC)
        grad_phi_BC = torch.autograd.grad(outputs=phi_pred_BC, inputs=grid_mesh_eval_xyz_BC,
                                         grad_outputs=torch.ones_like(phi_pred_BC),
                                         create_graph=True)[0]

        B_pred_BC = (-1 * grad_phi_BC).cpu().detach().numpy()
        labels = ['B$_{x}$', 'B$_{y}$', "B$_{z}$"]

        fig, ax = pl.subplots(1, 3, figsize=(10, 3))
        for i, el in enumerate(ax):
            im1 = el.imshow(B_pred_0[..., i].reshape(num_pts, num_pts),
                            cmap='seismic',
                            vmin=np.nanquantile(B_pred_0[..., i], 0.1),
                            vmax=np.nanquantile(B_pred_0[..., i], 0.9))
            el.set_title(labels[i])
            pl.colorbar(im1)
        fig.suptitle("Lunar surface estimate")
        pl.tight_layout()
        pl.savefig(f"eval_surface_{epoch:d}.png")
        pl.show()
        pl.close()

        fig, ax = pl.subplots(1, 3, figsize=(10, 3))
        for i, el in enumerate(ax):
            im1 = el.imshow(B_pred_BC[..., i].reshape(num_pts, num_pts),
                            cmap='seismic',
                            vmin=np.nanquantile(B_pred_BC[..., i], 0.1),
                            vmax=np.nanquantile(B_pred_BC[..., i], 0.9))
            pl.colorbar(im1)
            el.set_title(labels[i])

        pl.tight_layout()
        fig.suptitle("Lunar BC estimate")

        pl.savefig(f"eval_BC_{epoch:d}.png")
        pl.show()
        pl.close()

        args = [lunar_data.b_x[::10],
                lunar_data.b_y[::10],
                lunar_data.b_z[::10]]
        if (epoch == 0) or (epoch == 1):
            fig, ax = pl.subplots(1, 3, figsize=(10, 3))
            for i, el in enumerate(args):
                im1 = ax[i].scatter(lunar_data.phi[::10],
                                   lunar_data.theta[::10],
                                   c=el,
                                   cmap='seismic',
                                   vmin=np.nanquantile(el, 0.1),
                                   vmax=np.nanquantile(el, 0.9))
                pl.colorbar(im1)
                ax[i].set_title(labels[i])

            pl.tight_layout()
            fig.suptitle("Lunar true BC estimate")

            pl.savefig(f"true_BC_{epoch:d}.png")
            pl.show()
            pl.close()

    def evaluate_model(self, epoch, lunar_data):
        # Predict the potential and field after training
        self.plot_B_eval(epoch, lunar_data)