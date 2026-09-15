import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

class MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(2, 64)               #two inputs, 64 neurons
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 64)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(64, 1)               #one output

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x


  

#mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
#                    nn.Linear(64, 64), nn.ReLU(),
#                    nn.Linear(64, 1))
def cosine_lr_lambda(total_steps: int, warmup_frac: float = 0.08,
                     start_factor: float = 0.01, end_factor: float = 0.1):
    """LR multiplier: linear warmup eta/100 -> eta over ``warmup_frac`` of the steps,
    then cosine annealing eta -> eta/10. The per-cell cosine schedule used throughout."""
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return start_factor + (1.0 - start_factor) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return end_factor + (1.0 - end_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn

def train_flowmatching(mlp, lr, n_train_steps=1000, n_batch=256):
    # Define the loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda(n_train_steps))

    #trainings loop
    losses=[]
    for i in range(n_train_steps):

    
        #Triple generation (x_0, x_1, t) by drawing from different distributions
        x_0 = torch.randn(n_batch, 1) # ~ N(0,1)
        x_1 = 2*torch.randn(n_batch, 1) + 2 # ~ N(2,4)
        t = torch.rand(n_batch, 1)

        x_t = (1 - t)*x_0 + t*x_1
        u = x_1 - x_0

        optimizer.zero_grad()

        input = torch.cat((x_t, t), dim=1)       #we are training batches
        output = mlp(input)
        
    
        loss = loss_function(output, u)
        losses.append(loss.item())
 
        loss.backward()

        optimizer.step()
        scheduler.step()
    
    return mlp, losses

def plot_losses(all_losses, learning_rates, val_losses):
    fig = plt.figure(figsize=(14, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.1)

    # --- Linke Subfigure: Loss-Verlauf pro Trainingsschritt (wie bisher) ---
    ax1 = subfigs[0].subplots()
    for losses, lr in zip(all_losses, learning_rates):
        ax1.plot(losses, label=f"lr={lr:.1e}")
    ax1.set_yscale('log')
    ax1.set_xlabel("Trainingsschritt")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss-Verlauf für verschiedene Lernraten")
    ax1.legend()

    # --- Rechte Subfigure: Val-Loss + Parabel-Fit ---
    ax2 = subfigs[1].subplots()

    ax2.plot(learning_rates, val_losses, "o", color="steelblue", label="measured")
    ax2.set_xscale("log")
    ax2.set_xlabel("Learning rate")
    ax2.set_ylabel("Validation loss")
    ax2.set_title("Validation Loss vs. Lernrate")
    ax2.legend()

    plt.show()
    #return lr_star

@torch.no_grad()
def validation_loss(mlp):
    val_gen = torch.Generator().manual_seed(12876)
    x0_val = torch.randn(1000, 1, generator=val_gen)
    x1_val = 2+ 2*torch.randn(1000, 1, generator=val_gen)
    t = torch.rand(1000, 1, generator=val_gen)
    x_t = x0_val*(1-t) + x1_val*t
    u= x1_val -x0_val
    
    input = torch.cat((x_t, t),dim=1)
    output = mlp(input)
    u_mean = u.mean().item()
    u_scale = 1.0/u.std().item()
    
    target = (output - u_mean)*u_scale
        
    loss_function = torch.nn.MSELoss()
    val_loss = loss_function(output, target).item()
    return val_loss


def optimizing_learning_rate():
    losses=[]
    val_losses=[]
    
    learn_rates = np.geomspace(3e-4, 1e-1, 9)
         
    for lr in learn_rates:
        mlp = MLP()
        mlp, loss = train_flowmatching(mlp,lr)
        losses.append(loss)
        #loss_lr.append(losses[-1])
        val_losses.append(validation_loss(mlp))
    
    val_losses = np.array(val_losses)
    print(f"val_loss:{val_losses}")
    plot_losses(losses, learn_rates, val_losses)
    
            
        
optimizing_learning_rate()
        
@torch.no_grad()
def euler_sampler(mlp, x):
    n_steps = 100
    dt = 1/n_steps 
    traj = [x.clone()]

    for i in range(n_steps):
        t = torch.full((x.shape[0],1), i*dt)
        input = torch.cat((x, t), dim=1)
        v = mlp(input)
        x = x + v*dt
        traj.append(x.clone())
    
    return x, torch.cat(traj, dim=1)

#Daten erzeugen
mlp = MLP()
mlp, losses = train_flowmatching(mlp, 9e-2)

eval_gen = torch.Generator().manual_seed(349508)
x_normal = torch.randn(20000, 1, generator=eval_gen)
samples, traj = euler_sampler(mlp, x_normal)

#6. Visualizing the results


# Ein einziger Sampling-Aufruf liefert Endpunkte UND komplette Trajektorien
samples_np = samples.detach().numpy().flatten()

# Parameter der Zielverteilung N(2,4)
mu, sigma = 2, 2
x = np.linspace(-6, 10, 500)
pdf = norm.pdf(x, loc=mu, scale=sigma)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# --- Erster Plot: Histogramm + theoretische Gaußkurve ---
ax[0].hist(samples_np, bins=40, density=True, alpha=0.6,
           color="skyblue", edgecolor="black", label="Samples")
ax[0].plot(x, pdf, color="red", lw=2, label="N(2,4)")
ax[0].set_title("Histogramm und Gaußverteilung")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Dichte")
ax[0].legend()

# --- Zweiter Plot: Heatmap des gelernten Vektorfelds v_theta(x,t) ---
x_vals = torch.linspace(-3, 6, 200)
t_vals = torch.linspace(0, 1, 200)
X, T = torch.meshgrid(x_vals, t_vals, indexing='ij')

inp = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)
with torch.no_grad():
    V_flat = mlp(inp)
V_plot = V_flat.reshape(X.shape).numpy().T

im = ax[1].imshow(
    V_plot,
    extent=[x_vals.min(), x_vals.max(), t_vals.min(), t_vals.max()],
    origin='lower', aspect='auto', cmap='coolwarm'
)
fig.colorbar(im, ax=ax[1], label=r'$v_\theta(x,t)$')
ax[1].set_xlabel('x')
ax[1].set_ylabel('t')
ax[1].set_title('Gelerntes Vektorfeld')

# --- Trajektorien: Teilmenge der schon gezogenen Samples nehmen ---
n_traj = 20
idx = torch.linspace(0, x_normal.shape[0] - 1, n_traj).long()
traj_subset = traj[idx].detach().numpy()          # shape: (n_traj, n_steps+1)

n_steps = traj.shape[1] - 1
t_axis = np.linspace(0, 1, n_steps + 1)

for i in range(traj_subset.shape[0]):
    ax[1].plot(traj_subset[i], t_axis, color='black', alpha=0.7, linewidth=1)

# --- Randverteilungen bei t=0 (N(0,1)) und t=1 (N(2,4)) einzeichnen ---
x_plot = np.linspace(x_vals.min().item(), x_vals.max().item(), 300)

pdf0 = norm.pdf(x_plot, loc=0, scale=1)     # Startverteilung N(0,1)
pdf1 = norm.pdf(x_plot, loc=2, scale=2)     # Zielverteilung N(2,4)

band_height = 0.12
pdf0_scaled = pdf0 / pdf0.max() * band_height
pdf1_scaled = pdf1 / pdf1.max() * band_height

ax[1].set_ylim(-band_height - 0.02, 1 + band_height + 0.02)

# Kurve unten (t=0), nach unten wachsend
ax[1].fill_between(x_plot, 0, -pdf0_scaled, color='steelblue', alpha=0.6, zorder=3)
ax[1].plot(x_plot, -pdf0_scaled, color='steelblue', linewidth=1.2, zorder=3)
ax[1].axhline(0, color='black', linewidth=0.8, zorder=3)

# Kurve oben (t=1), nach oben wachsend
ax[1].fill_between(x_plot, 1, 1 + pdf1_scaled, color='indianred', alpha=0.6, zorder=3)
ax[1].plot(x_plot, 1 + pdf1_scaled, color='indianred', linewidth=1.2, zorder=3)
ax[1].axhline(1, color='black', linewidth=0.8, zorder=3)

ax[1].text(x_vals.max().item(), -band_height * 0.6, 'N(0,1)', color='steelblue',
           ha='right', va='center', fontsize=8)
ax[1].text(x_vals.max().item(), 1 + band_height * 0.6, 'N(2,4)', color='indianred',
           ha='right', va='center', fontsize=8)

ax[1].secondary_xaxis('top').tick_params(direction='in', length=4, labelsize=8)

plt.tight_layout()
plt.show()