# Flow Matching & Scaling Laws

This repository contains a **1D flow matching demo** and an attempt to apply **scaling-law analysis** (learning-rate tuning) to it. It also includes scripts that reproduce Matthias's scaling-law experiments, which served as the reference for this analysis.

## Repository structure

```
.
├── Flow_Matching/
│   ├── Flow_Matching.ipynb      # Main notebook: experiments, final model, visualizations
│   ├── functions.py             # Core logic: model, training, validation, sampling, optimazation
│   └── plot.py                  # All plotting / visualization functions
├── Scaling_Laws/
│   ├── Scaling_Laws_Reproduction.ipynb
│   └── Scaling_Laws_Reproduction_scheduler_exp.ipynb
├── .gitattributes
└── .gitignore
```

## Flow matching demo (`Flow_Matching/`)

The demo learns a velocity field that transports a source distribution to a target distribution in one dimension:

- **Source:** `x_0 ~ N(0, 1)`
- **Target:** `x_1 ~ N(2, 4)` (mean 2, standard deviation 2)
- **Interpolation:** `x_t = (1 - t) * x_0 + t * x_1`, with target velocity `u = x_1 - x_0`
- **Model:** MLP with input `(x_t, t)`, two hidden layers of 64 units (ReLU), one output
- **Training:** MSE loss, AdamW, cosine learning-rate schedule with linear warmup
- **Sampling:** Euler integration of the learned velocity field (100 steps)

### `functions.py` (logic)

| Function | Purpose |
|---|---|
| `MLP` | The velocity-field network |
| `cosine_lr_lambda` | Learning-rate schedule (warmup, then cosine annealing) |
| `train_flowmatching` | Trains the network with flow matching |
| `validation_loss` | Validation loss on a fixed validation set (own generator, independent of the global seed) |
| `optimizing_learning_rate_seeds` | Learning-rate sweep; for each learning rate, trains several networks with different random seeds and returns mean and standard deviation of the validation loss |
| `euler_sampler` | Generates samples and trajectories by Euler integration |

### `plot.py` (visualization)

| Function | Purpose |
|---|---|
| `plot_stacked_losses_errorbar` | Stacked plots for different learning-rate ranges, with error bars (mean ± std over runs) and the number of training steps in the title |
| `plot_samples_n_trajectories` | Histogram of generated samples against the target density, plus the learned vector field with sampled trajectories |

### `Flow_Matching.ipynb` (workflow)

1. **Optimizing the learning rate:** sweep over several learning-rate ranges, averaged over multiple runs.
2. **Training the final network and sampling:** train with the chosen learning rate, sample from `N(0, 1)`, and compare the result to `N(2, 4)`.
3. **Visualizing:** validation loss vs. learning rate with error bars.

### Notes on the uncertainty estimate

Error bars show the standard deviation over runs with different random seeds. The seed changes both the weight initialization and the training batches. The validation set is fixed, so the spread does not contain measurement noise.

## Scaling-law reproduction (`Scaling_Laws/`)

Notebooks that reproduce Matthias's scaling-law experiments: a teacher-student setup where a student MLP is trained on the outputs of a fixed random teacher MLP, and the optimal learning rate is found by fitting a parabola to the validation loss in `log10(learning rate)`.
In 'Scaling_Laws_Reproduction_scheduler_exp' 

## Setup

```bash
pip install numpy torch matplotlib scipy
```

Then open `Flow_Matching/Flow_Matching.ipynb` in Jupyter or VS Code and run the cells from top to bottom.
