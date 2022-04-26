import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T

from matplotlib import pyplot as plt

import os
import argparse

import models


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument('--max-epoch', default=50, type=int, help="Max training epochs")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data...')
transform_train = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

transform_test = T.Compose(
    [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
)

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
train_batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=train_batch_size, shuffle=True, num_workers=2
)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False, num_workers=2
)

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

# Model
print(f"==> Using model ResNet18")
net = models.ResNet18()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    cudnn.deterministic = True
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
torch.save(net.state_dict(), './checkpoints/resnet18.pth')

net_mse = models.ResNet18()
net_bce = models.ResNet18()
net_ce = models.ResNet18()
net_hb = models.ResNet18()

if device == 'cuda':
    net_mse = nn.DataParallel(net_mse)
    net_bce = nn.DataParallel(net_bce)
    net_ce = nn.DataParallel(net_ce)
    net_hb = nn.DataParallel(net_hb)

net_mse.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_bce.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_ce.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_hb.load_state_dict(torch.load('./checkpoints/resnet18.pth'))

criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
criterion_ce = nn.CrossEntropyLoss()
criterion_hb = nn.HuberLoss()
# Unlike 'mean', 'batchmean' divides by the batch size, and aligns with the KL div
# criterion_kl = nn.HuberLoss(reduction='batchmean')  # KL loss goes to negative...

lr = 0.001
optimizer_mse = optim.Adam(net_mse.parameters(), lr=lr, weight_decay=5e-4)
optimizer_bce = optim.Adam(net_bce.parameters(), lr=lr, weight_decay=5e-4)
optimizer_ce = optim.Adam(net_ce.parameters(), lr=lr, weight_decay=5e-4)
optimizer_hb = optim.Adam(net_hb.parameters(), lr=lr, weight_decay=5e-4)

scheduler_mse = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mse, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_bce = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_bce, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_ce = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ce, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_hb = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_hb, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)


# Training
def train(epoch):
    net_mse.train()
    net_bce.train()
    net_ce.train()
    net_hb.train()

    train_loss_mse = 0
    train_loss_bce = 0
    train_loss_ce = 0
    train_loss_hb = 0

    correct_mse = 0
    correct_bce = 0
    correct_ce = 0
    correct_hb = 0

    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets_onehot = F.one_hot(targets, num_classes=10).to(torch.float)

        optimizer_mse.zero_grad()
        optimizer_bce.zero_grad()
        optimizer_ce.zero_grad()
        optimizer_hb.zero_grad()

        outputs_mse = net_mse(inputs)
        loss_mse = criterion_mse(outputs_mse, targets_onehot)
        loss_mse.backward()
        optimizer_mse.step()
        train_loss_mse += loss_mse.item()
        _, predicted_mse = outputs_mse.max(1)
        correct_mse += predicted_mse.eq(targets).sum().item()

        outputs_bce = net_bce(inputs)
        loss_bce = criterion_bce(outputs_bce, targets_onehot)
        loss_bce.backward()
        optimizer_bce.step()
        train_loss_bce += loss_bce.item()
        _, predicted_bce = outputs_bce.max(1)
        correct_bce += predicted_bce.eq(targets).sum().item()

        outputs_ce = net_ce(inputs)
        loss_ce = criterion_ce(outputs_ce, targets)
        loss_ce.backward()
        optimizer_ce.step()
        train_loss_ce += loss_ce.item()
        _, predicted_ce = outputs_ce.max(1)
        correct_ce += predicted_ce.eq(targets).sum().item()

        outputs_hb = net_hb(inputs)
        loss_hb = criterion_hb(outputs_hb, targets_onehot)
        loss_hb.backward()
        optimizer_hb.step()
        train_loss_hb += loss_hb.item()
        _, predicted_hb = outputs_hb.max(1)
        correct_hb += predicted_hb.eq(targets).sum().item()

        total += targets.size(0)

    epoch_loss_mse = train_loss_mse / ((batch_idx + 1) * train_batch_size)
    epoch_acc_mse = 100.0 * correct_mse / total

    epoch_loss_bce = train_loss_bce / ((batch_idx + 1) * train_batch_size)
    epoch_acc_bce = 100.0 * correct_bce / total

    epoch_loss_ce = train_loss_ce / ((batch_idx + 1) * train_batch_size)
    epoch_acc_ce = 100.0 * correct_ce / total

    epoch_loss_hb = train_loss_hb / ((batch_idx + 1) * train_batch_size)
    epoch_acc_hb = 100.0 * correct_hb / total

    return (
        epoch_loss_mse,
        epoch_acc_mse,
        epoch_loss_bce,
        epoch_acc_bce,
        epoch_loss_ce,
        epoch_acc_ce,
        epoch_loss_hb,
        epoch_acc_hb,
    )


def test(epoch):
    net_mse.eval()
    net_bce.eval()
    net_ce.eval()
    net_hb.eval()

    test_loss_mse = 0
    test_loss_bce = 0
    test_loss_ce = 0
    test_loss_hb = 0

    correct_mse = 0
    correct_bce = 0
    correct_ce = 0
    correct_hb = 0

    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets_onehot = F.one_hot(targets, num_classes=10).to(torch.float)

            outputs_mse = net_mse(inputs)
            loss_mse = criterion_mse(outputs_mse, targets_onehot)
            test_loss_mse += loss_mse.item()
            _, predicted_mse = outputs_mse.max(1)
            correct_mse += predicted_mse.eq(targets).sum().item()

            outputs_bce = net_bce(inputs)
            loss_bce = criterion_bce(outputs_bce, targets_onehot)
            test_loss_bce += loss_bce.item()
            _, predicted_bce = outputs_bce.max(1)
            correct_bce += predicted_bce.eq(targets).sum().item()

            outputs_ce = net_ce(inputs)
            loss_ce = criterion_ce(outputs_ce, targets)
            test_loss_ce += loss_ce.item()
            _, predicted_ce = outputs_ce.max(1)
            correct_ce += predicted_ce.eq(targets).sum().item()

            outputs_hb = net_hb(inputs)
            loss_hb = criterion_hb(outputs_hb, targets_onehot)
            test_loss_hb += loss_hb.item()
            _, predicted_hb = outputs_hb.max(1)
            correct_hb += predicted_hb.eq(targets).sum().item()

            total += targets.size(0)

    epoch_loss_mse = test_loss_mse / ((batch_idx + 1) * test_batch_size)
    epoch_acc_mse = 100.0 * correct_mse / total

    epoch_loss_bce = test_loss_bce / ((batch_idx + 1) * test_batch_size)
    epoch_acc_bce = 100.0 * correct_bce / total

    epoch_loss_ce = test_loss_ce / ((batch_idx + 1) * test_batch_size)
    epoch_acc_ce = 100.0 * correct_ce / total

    epoch_loss_hb = test_loss_hb / ((batch_idx + 1) * test_batch_size)
    epoch_acc_hb = 100.0 * correct_hb / total

    return (
        epoch_loss_mse,
        epoch_acc_mse,
        epoch_loss_bce,
        epoch_acc_bce,
        epoch_loss_ce,
        epoch_acc_ce,
        epoch_loss_hb,
        epoch_acc_hb,
    )


# Training
train_losses_mse = []
train_losses_bce = []
train_losses_ce = []
train_losses_hb = []

train_accuracies_mse = []
train_accuracies_bce = []
train_accuracies_ce = []
train_accuracies_hb = []

test_losses_mse = []
test_losses_bce = []
test_losses_ce = []
test_losses_hb = []

test_accuracies_mse = []
test_accuracies_bce = []
test_accuracies_ce = []
test_accuracies_hb = []

for epoch in range(args.max_epoch):
    print(f"Epoch: {epoch + 1}")

    (
        train_loss_mse,
        train_acc_mse,
        train_loss_bce,
        train_acc_bce,
        train_loss_ce,
        train_acc_ce,
        train_loss_hb,
        train_acc_hb,
    ) = train(epoch)

    (
        test_loss_mse,
        test_acc_mse,
        test_loss_bce,
        test_acc_bce,
        test_loss_ce,
        test_acc_ce,
        test_loss_hb,
        test_acc_hb,
    ) = test(epoch)

    scheduler_mse.step(test_loss_mse)
    scheduler_bce.step(test_loss_bce)
    scheduler_ce.step(test_loss_ce)
    scheduler_hb.step(test_loss_hb)

    # Logging
    train_losses_mse.append(train_loss_mse)
    train_losses_bce.append(train_loss_bce)
    train_losses_ce.append(train_loss_ce)
    train_losses_hb.append(train_loss_hb)

    train_accuracies_mse.append(train_acc_mse)
    train_accuracies_bce.append(train_acc_bce)
    train_accuracies_ce.append(train_acc_ce)
    train_accuracies_hb.append(train_acc_hb)

    test_losses_mse.append(test_loss_mse)
    test_losses_bce.append(test_loss_bce)
    test_losses_ce.append(test_loss_ce)
    test_losses_hb.append(test_loss_hb)

    test_accuracies_mse.append(test_acc_mse)
    test_accuracies_bce.append(test_acc_bce)
    test_accuracies_ce.append(test_acc_ce)
    test_accuracies_hb.append(test_acc_hb)


# Plot
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses_mse, label="MSE")
plt.plot(train_losses_bce, label="BCE")
plt.plot(train_losses_ce, label="CrossEntropy")
plt.plot(train_losses_hb, label="Huber")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(train_accuracies_mse, label="MSE")
plt.plot(train_accuracies_bce, label="BCE")
plt.plot(train_accuracies_ce, label="CrossEntropy")
plt.plot(train_accuracies_hb, label="Huber")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(test_losses_mse, label="MSE")
plt.plot(test_losses_bce, label="BCE")
plt.plot(test_losses_ce, label="CrossEntropy")
plt.plot(test_losses_hb, label="Huber")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(test_accuracies_mse, label="MSE")
plt.plot(test_accuracies_bce, label="BCE")
plt.plot(test_accuracies_ce, label="CrossEntropy")
plt.plot(test_accuracies_hb, label="Huber")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

if not os.path.isdir('log'):
    os.mkdir('log')
plt.savefig(f'./log/criterion.png')

print("Done")
