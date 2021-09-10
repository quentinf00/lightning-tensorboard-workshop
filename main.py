# %% Imports
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

# %% Download data
class L63():
    def __init__(self, sigma=10, rho=28, beta=8/3, init=(1, 10, 20), dt=1e-2):
        self.sigma, self.rho, self.beta = sigma, rho, beta 
        self.x, self.y, self.z = init
        self.dt = dt
        self.ts = [0]
        self.t = 0
        self.hist = [init]
    
    def step(self):
        self.x += (self.sigma * (self.y - self.x)) * self.dt
        self.y += (self.x * (self.rho - self.z)) * self.dt
        self.z += (self.x * self.y - self.beta * self.z) * self.dt
        self.hist.append([self.x, self.y, self.z])
        self.t += self.dt
        self.ts.append(self.t)
    
    def integrate(self, n_steps):
        for n in range(n_steps): self.step()
        return self

n_points = 1000
data = torch.tensor(L63().integrate(n_steps=n_points).hist)

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
ax.plot(data[:, 0], data[:, 1], data[:, 2], '+')

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9, 9))
ax1.plot(data[:, 0])
ax2.plot(data[:, 1])
ax3.plot(data[:, 2])

# %% Define task
class LitMod(pl.LightningModule):
    def __init__(self,
            inp_data,
            out_data,
            split=(500, 200, 300),
            batch_size=20,
            hidden=(16, 16, 16),
            act='relu',
        ):
        super().__init__()
        # Data
        self.data = data
        self.train_ds, self.val_ds, self.test_ds = (
                torch.utils.data.random_split(
                    torch.utils.data.TensorDataset(inp_data, out_data),
                    split,
                )
        )
        self.batch_size = batch_size

        # Model
        dims = [1] + list(hidden) 
        act = torch.nn.ReLU()
        self.mod = torch.nn.Sequential(
                *[
                    *(torch.nn.Linear(in_features=inp_f, out_features=out_f), act)
                    for inp_f, out_f in zip(dims[:-1], dims[1:])
                ],
                torch.nn.Linear(dims[-1], 3)

        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)

    def forward(self, x):
        return 
    def training_step(self, batch):
        x, y = batch
        out =  self(x)
        
        loss = (
            (out - y)**2
        ).mean()

        return loss

    def validation_step()

# %% Define task
test, (train, val) = ...



