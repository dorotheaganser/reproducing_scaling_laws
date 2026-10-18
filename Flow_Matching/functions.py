import torch
import torch.nn as nn
import numpy as np
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
        
    loss_function = torch.nn.MSELoss()
    val_loss = loss_function(output, u).item()
    return val_loss



def optimizing_learning_rate(learn_rates):
    losses=[]
    val_losses=[]
    
         
    for lr in learn_rates:
        mlp = MLP()
        mlp, loss = train_flowmatching(mlp,lr)
        losses.append(loss)
        #loss_lr.append(losses[-1])
        val_losses.append(validation_loss(mlp))

    losses = np.array(losses)
    val_losses = np.array(val_losses)

    return losses, val_losses
    
            
    losses=[]

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
