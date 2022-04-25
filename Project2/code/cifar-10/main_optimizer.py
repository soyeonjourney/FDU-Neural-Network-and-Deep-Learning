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

net_sgd = models.ResNet18()
net_adagrad = models.ResNet18()
net_adadelta = models.ResNet18()
net_adam = models.ResNet18()

if device == 'cuda':
    net_sgd = nn.DataParallel(net_sgd)
    net_adagrad = nn.DataParallel(net_adagrad)
    net_adadelta = nn.DataParallel(net_adadelta)
    net_adam = nn.DataParallel(net_adam)

net_sgd.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_adagrad.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_adadelta.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_adam.load_state_dict(torch.load('./checkpoints/resnet18.pth'))


criterion = nn.CrossEntropyLoss()

lr, lr_adam = 0.1, 0.001
optimizer_sgd = optim.SGD(net_sgd.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_adagrad = optim.Adagrad(net_adagrad.parameters(), lr=lr, weight_decay=5e-4)
optimizer_adadelta = optim.Adadelta(net_adadelta.parameters(), lr=lr, weight_decay=5e-4)
optimizer_adam = optim.Adam(net_adam.parameters(), lr=lr_adam, weight_decay=5e-4)

scheduler_sgd = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_sgd, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_adagrad = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adagrad, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_adadelta = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adadelta, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adam, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)


# Training
def train(epoch):
    net_sgd.train()
    net_adagrad.train()
    net_adadelta.train()
    net_adam.train()

    train_loss_sgd = 0
    train_loss_adagrad = 0
    train_loss_adadelta = 0
    train_loss_adam = 0

    correct_sgd = 0
    correct_adagrad = 0
    correct_adadelta = 0
    correct_adam = 0

    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_sgd.zero_grad()
        optimizer_adagrad.zero_grad()
        optimizer_adadelta.zero_grad()
        optimizer_adam.zero_grad()

        outputs_sgd = net_sgd(inputs)
        loss_sgd = criterion(outputs_sgd, targets)
        loss_sgd.backward()
        optimizer_sgd.step()
        train_loss_sgd += loss_sgd.item()
        _, predicted_sgd = outputs_sgd.max(1)
        correct_sgd += predicted_sgd.eq(targets).sum().item()

        outputs_adagrad = net_adagrad(inputs)
        loss_adagrad = criterion(outputs_adagrad, targets)
        loss_adagrad.backward()
        optimizer_adagrad.step()
        train_loss_adagrad += loss_adagrad.item()
        _, predicted_adagrad = outputs_adagrad.max(1)
        correct_adagrad += predicted_adagrad.eq(targets).sum().item()

        outputs_adadelta = net_adadelta(inputs)
        loss_adadelta = criterion(outputs_adadelta, targets)
        loss_adadelta.backward()
        optimizer_adadelta.step()
        train_loss_adadelta += loss_adadelta.item()
        _, predicted_adadelta = outputs_adadelta.max(1)
        correct_adadelta += predicted_adadelta.eq(targets).sum().item()

        outputs_adam = net_adam(inputs)
        loss_adam = criterion(outputs_adam, targets)
        loss_adam.backward()
        optimizer_adam.step()
        train_loss_adam += loss_adam.item()
        _, predicted_adam = outputs_adam.max(1)
        correct_adam += predicted_adam.eq(targets).sum().item()

        total += targets.size(0)

    epoch_loss_sgd = train_loss_sgd / ((batch_idx + 1) * train_batch_size)
    epoch_acc_sgd = 100.0 * correct_sgd / total

    epoch_loss_adagrad = train_loss_adagrad / ((batch_idx + 1) * train_batch_size)
    epoch_acc_adagrad = 100.0 * correct_adagrad / total

    epoch_loss_adadelta = train_loss_adadelta / ((batch_idx + 1) * train_batch_size)
    epoch_acc_adadelta = 100.0 * correct_adadelta / total

    epoch_loss_adam = train_loss_adam / ((batch_idx + 1) * train_batch_size)
    epoch_acc_adam = 100.0 * correct_adam / total

    return (
        epoch_loss_sgd,
        epoch_acc_sgd,
        epoch_loss_adagrad,
        epoch_acc_adagrad,
        epoch_loss_adadelta,
        epoch_acc_adadelta,
        epoch_loss_adam,
        epoch_acc_adam,
    )


def test(epoch):
    net_sgd.eval()
    net_adagrad.eval()
    net_adadelta.eval()
    net_adam.eval()

    test_loss_sgd = 0
    test_loss_adagrad = 0
    test_loss_adadelta = 0
    test_loss_adam = 0

    correct_sgd = 0
    correct_adagrad = 0
    correct_adadelta = 0
    correct_adam = 0

    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs_sgd = net_sgd(inputs)
            loss_sgd = criterion(outputs_sgd, targets)
            test_loss_sgd += loss_sgd.item()
            _, predicted_sgd = outputs_sgd.max(1)
            correct_sgd += predicted_sgd.eq(targets).sum().item()

            outputs_adagrad = net_adagrad(inputs)
            loss_adagrad = criterion(outputs_adagrad, targets)
            test_loss_adagrad += loss_adagrad.item()
            _, predicted_adagrad = outputs_adagrad.max(1)
            correct_adagrad += predicted_adagrad.eq(targets).sum().item()

            outputs_adadelta = net_adadelta(inputs)
            loss_adadelta = criterion(outputs_adadelta, targets)
            test_loss_adadelta += loss_adadelta.item()
            _, predicted_adadelta = outputs_adadelta.max(1)
            correct_adadelta += predicted_adadelta.eq(targets).sum().item()

            outputs_adam = net_adam(inputs)
            loss_adam = criterion(outputs_adam, targets)
            test_loss_adam += loss_adam.item()
            _, predicted_adam = outputs_adam.max(1)
            correct_adam += predicted_adam.eq(targets).sum().item()

            total += targets.size(0)

    epoch_loss_sgd = test_loss_sgd / ((batch_idx + 1) * test_batch_size)
    epoch_acc_sgd = 100.0 * correct_sgd / total

    epoch_loss_adagrad = test_loss_adagrad / ((batch_idx + 1) * test_batch_size)
    epoch_acc_adagrad = 100.0 * correct_adagrad / total

    epoch_loss_adadelta = test_loss_adadelta / ((batch_idx + 1) * test_batch_size)
    epoch_acc_adadelta = 100.0 * correct_adadelta / total

    epoch_loss_adam = test_loss_adam / ((batch_idx + 1) * test_batch_size)
    epoch_acc_adam = 100.0 * correct_adam / total

    return (
        epoch_loss_sgd,
        epoch_acc_sgd,
        epoch_loss_adagrad,
        epoch_acc_adagrad,
        epoch_loss_adadelta,
        epoch_acc_adadelta,
        epoch_loss_adam,
        epoch_acc_adam,
    )


# Training
train_losses_sgd = []
train_losses_adagrad = []
train_losses_adadelta = []
train_losses_adam = []

train_accuracies_sgd = []
train_accuracies_adagrad = []
train_accuracies_adadelta = []
train_accuracies_adam = []

test_losses_sgd = []
test_losses_adagrad = []
test_losses_adadelta = []
test_losses_adam = []

test_accuracies_sgd = []
test_accuracies_adagrad = []
test_accuracies_adadelta = []
test_accuracies_adam = []

for epoch in range(args.max_epoch):
    print(f"Epoch: {epoch + 1}")

    (
        train_loss_sgd,
        train_acc_sgd,
        train_loss_adagrad,
        train_acc_adagrad,
        train_loss_adadelta,
        train_acc_adadelta,
        train_loss_adam,
        train_acc_adam,
    ) = train(epoch)

    (
        test_loss_sgd,
        test_acc_sgd,
        test_loss_adagrad,
        test_acc_adagrad,
        test_loss_adadelta,
        test_acc_adadelta,
        test_loss_adam,
        test_acc_adam,
    ) = test(epoch)

    scheduler_sgd.step(test_loss_sgd)
    scheduler_adagrad.step(test_loss_adagrad)
    scheduler_adadelta.step(test_loss_adadelta)
    scheduler_adam.step(test_loss_adam)

    # Logging
    train_losses_sgd.append(train_loss_sgd)
    train_losses_adagrad.append(train_loss_adagrad)
    train_losses_adadelta.append(train_loss_adadelta)
    train_losses_adam.append(train_loss_adam)

    train_accuracies_sgd.append(train_acc_sgd)
    train_accuracies_adagrad.append(train_acc_adagrad)
    train_accuracies_adadelta.append(train_acc_adadelta)
    train_accuracies_adam.append(train_acc_adam)

    test_losses_sgd.append(test_loss_sgd)
    test_losses_adagrad.append(test_loss_adagrad)
    test_losses_adadelta.append(test_loss_adadelta)
    test_losses_adam.append(test_loss_adam)

    test_accuracies_sgd.append(test_acc_sgd)
    test_accuracies_adagrad.append(test_acc_adagrad)
    test_accuracies_adadelta.append(test_acc_adadelta)
    test_accuracies_adam.append(test_acc_adam)


# Plot
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses_sgd, label="SGD")
plt.plot(train_losses_adagrad, label="Adagrad")
plt.plot(train_losses_adadelta, label="Adadelta")
plt.plot(train_losses_adam, label="Adam")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(train_accuracies_sgd, label="SGD")
plt.plot(train_accuracies_adagrad, label="Adagrad")
plt.plot(train_accuracies_adadelta, label="Adadelta")
plt.plot(train_accuracies_adam, label="Adam")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(test_losses_sgd, label="SGD")
plt.plot(test_losses_adagrad, label="Adagrad")
plt.plot(test_losses_adadelta, label="Adadelta")
plt.plot(test_losses_adam, label="Adam")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(test_accuracies_sgd, label="SGD")
plt.plot(test_accuracies_adagrad, label="Adagrad")
plt.plot(test_accuracies_adadelta, label="Adadelta")
plt.plot(test_accuracies_adam, label="Adam")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

if not os.path.isdir('log'):
    os.mkdir('log')
plt.savefig(f'./log/optimizer.png')

print("Done")
