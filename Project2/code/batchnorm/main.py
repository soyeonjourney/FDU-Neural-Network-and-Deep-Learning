import os
import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt

from models.vgg import VGG_A, VGG_A_BatchNorm
from utils.loaders import get_cifar_loader

matplotlib.use('Agg')


# Add dir to path
figures_path = './log'
models_path = './checkpoints'
if not os.path.isdir(figures_path):
    os.mkdir(figures_path)
if not os.path.isdir(models_path):
    os.mkdir(models_path)

# Make sure you are using the right device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=42, device='cpu'):
    torch.manual_seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate(model, val_loader):
    # Validation
    model.eval()

    total, correct = 0, 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    scheduler=None,
    num_epochs=100,
):
    # Logging initialization
    train_accuracies = []
    val_accuracies = []
    losses = []  # Training loss
    dists = []  # Distance between two gradients
    betas = []  # Beta smoothness
    grad_pre = None  # Previous gradient

    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        # Training
        model.train()

        if scheduler is not None:
            scheduler.step()

        total, correct = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Epoch logging
            losses.append(loss.item())
            grad = model.classifier[-1].weight.grad.detach().clone()
            if grad_pre is not None:
                dist = torch.dist(grad, grad_pre).item()
                dists.append(dist)
                beta = dist / (optimizer.param_groups[0]['lr'] * torch.norm(grad_pre))
                betas.append(beta)
            grad_pre = grad

        # Training accuracy
        train_accuracies.append(correct / total)

        # Validation accuracy
        val_accuracy = validate(model, val_loader)
        val_accuracies.append(val_accuracy)

    return train_accuracies, val_accuracies, losses, dists, betas


# Use this function to plot the final grad dist landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_landscape(
    vgg_max, vgg_min, vggbn_max, vggbn_min, xlabel, ylabel, title, interval=50
):
    steps = np.arange(0, len(vgg_max), interval)
    vgg_max = vgg_max[steps]
    vgg_min = vgg_min[steps]
    vggbn_max = vggbn_max[steps]
    vggbn_min = vggbn_min[steps]

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 9))
    plt.plot(steps, vgg_max, color='mediumturquoise', alpha=0.5)
    plt.plot(steps, vgg_min, color='mediumturquoise', alpha=0.5)
    plt.fill_between(
        steps, vgg_min, vgg_max, color='mediumturquoise', alpha=0.5, label='VGG'
    )
    plt.plot(steps, vggbn_max, color='coral', alpha=0.5)
    plt.plot(steps, vggbn_min, color='coral', alpha=0.5)
    plt.fill_between(
        steps, vggbn_min, vggbn_max, color='coral', alpha=0.5, label='VGG + BN'
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(figures_path, f'{ylabel.lower()}-landscape.png'))


max_epochs = 20
lrs = [1e-3, 5e-4, 1e-4]
set_random_seeds(seed_value=42, device=device)

vgg_train, vgg_val, vgg_losses, vgg_dists, vgg_betas = [], [], [], [], []
vggbn_train, vggbn_val, vggbn_losses, vggbn_dists, vggbn_betas = [], [], [], [], []

# VGG-A
for lr in lrs:
    print("======================VGG-A======================")
    print(f"Learning rate: {lr}")
    model = VGG_A()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_acc, val_acc, losses, dists, betas = train(
        model, optimizer, criterion, train_loader, val_loader, num_epochs=max_epochs
    )
    vgg_train.append(train_acc)
    vgg_val.append(val_acc)
    vgg_losses.append(losses)
    vgg_dists.append(dists)
    vgg_betas.append(betas)

# VGG-A-BatchNorm
for lr in lrs:
    print("=================VGG-A-BarchNorm=================")
    print(f"Learning rate: {lr}")
    model = VGG_A_BatchNorm()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_acc, val_acc, losses, dists, betas = train(
        model, optimizer, criterion, train_loader, val_loader, num_epochs=max_epochs
    )
    vggbn_train.append(train_acc)
    vggbn_val.append(val_acc)
    vggbn_losses.append(losses)
    vggbn_dists.append(dists)
    vggbn_betas.append(betas)

# Maintain two lists: max_curve and min_curve, select the maximum value of loss in all
# models on the same step, add it to max_curve, and the minimum value to min_curve
vgg_train, vggbn_train = np.array(vgg_train), np.array(vggbn_train)
vgg_val, vggbn_val = np.array(vgg_val), np.array(vggbn_val)
vgg_losses, vggbn_losses = np.array(vgg_losses), np.array(vggbn_losses)
vgg_dists, vggbn_dists = np.array(vgg_dists), np.array(vggbn_dists)
vgg_betas, vggbn_betas = np.array(vgg_betas), np.array(vggbn_betas)

# Plot the loss landscape
vgg_max = np.max(vgg_losses, axis=0)
vgg_min = np.min(vgg_losses, axis=0)
vggbn_max = np.max(vggbn_losses, axis=0)
vggbn_min = np.min(vggbn_losses, axis=0)

plot_landscape(
    vgg_max=vgg_max,
    vgg_min=vgg_min,
    vggbn_max=vggbn_max,
    vggbn_min=vggbn_min,
    xlabel='Step',
    ylabel='Loss',
    title='Loss Landscape',
)

# Plot the dist landscape
vgg_max = np.max(vgg_dists, axis=0)
vgg_min = np.min(vgg_dists, axis=0)
vggbn_max = np.max(vggbn_dists, axis=0)
vggbn_min = np.min(vggbn_dists, axis=0)

plot_landscape(
    vgg_max=vgg_max,
    vgg_min=vgg_min,
    vggbn_max=vggbn_max,
    vggbn_min=vggbn_min,
    xlabel='Step',
    ylabel='GradDist',
    title='Grad Dist Landscape',
)

# Plot the beta landscape
vgg_max = np.max(vgg_betas, axis=0)
vggbn_max = np.max(vggbn_betas, axis=0)

steps = np.arange(0, len(vgg_max), 100)
vgg_max = vgg_max[steps]
vggbn_max = vggbn_max[steps]
plt.figure(figsize=(12, 9))
plt.plot(steps, vgg_max, color='mediumturquoise', label='VGG')
plt.plot(steps, vggbn_max, color='coral', label='VGG + BN')
plt.xlabel('Step')
plt.ylabel('Beta')
plt.legend()
plt.title('Beta Landscape')
plt.savefig(os.path.join(figures_path, 'beta-smooth.png'))
