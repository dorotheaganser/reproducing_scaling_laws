import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_stacked_losses_errorbar(learn_rates_list, val_losses_list, val_stds_list=None,
                        titles=None, figsize=(10, 6),
                        sharex=False, sharey=False, n_train_steps=None, n_seeds=None):
    #für drei gestackte plots brauchen wir dann wieder figsize=(5, 7), for one plot figsize=(10,6)
    n = len(learn_rates_list)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=sharex, sharey=sharey)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        yerr = None if val_stds_list is None else val_stds_list[i]
        ax.errorbar(learn_rates_list[i], val_losses_list[i], yerr=yerr,
                    fmt="o", color="steelblue", markersize=5,
                    capsize=3, label="mean ± std")
        ax.set_xscale("log")
        ax.set_ylabel("Validation loss")
        ax.legend()
        if titles is not None:
            ax.set_title(titles[i])

    axes[-1].set_xlabel("Learning rate")

    fig.tight_layout(rect=[0, 0, 1, 0.96])   # oberen 4 % für den Supertitle freihalten

    info = []
    if n_train_steps is not None:
        info.append(f" flowmatching: {n_train_steps} training steps")
    if n_seeds is not None:
        info.append(f"{n_seeds} seeds")
    if info:
        fig.suptitle(", ".join(info), fontsize=10)

    plt.show()
    return fig, axes
'''
Visualizing the results
Plotting Histogramm of end distribution and plot trajectories of start to end distribution
'''

def plot_samples_n_trajectories(mlp, samples, traj, x_normal):
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