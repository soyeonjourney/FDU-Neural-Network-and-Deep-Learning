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

import models_activation as models


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
net = models.ResNet18_ReLU()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    cudnn.deterministic = True
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
torch.save(net.state_dict(), './checkpoints/resnet18.pth')

net_relu = models.ResNet18_ReLU()
net_lrelu = models.ResNet18_LeakyReLU()
net_elu = models.ResNet18_ELU()
net_gelu = models.ResNet18_GELU()
net_sigmoid = models.ResNet18_Sigmoid()

if device == 'cuda':
    net_relu = nn.DataParallel(net_relu)
    net_lrelu = nn.DataParallel(net_lrelu)
    net_elu = nn.DataParallel(net_elu)
    net_gelu = nn.DataParallel(net_gelu)
    net_sigmoid = nn.DataParallel(net_sigmoid)

net_relu.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_lrelu.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_elu.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_gelu.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
net_sigmoid.load_state_dict(torch.load('./checkpoints/resnet18.pth'))


criterion = nn.CrossEntropyLoss()

lr = 0.001
optimizer_relu = optim.Adam(net_relu.parameters(), lr=lr, weight_decay=5e-4)
optimizer_lrelu = optim.Adam(net_lrelu.parameters(), lr=lr, weight_decay=5e-4)
optimizer_elu = optim.Adam(net_elu.parameters(), lr=lr, weight_decay=5e-4)
optimizer_gelu = optim.Adam(net_gelu.parameters(), lr=lr, weight_decay=5e-4)
optimizer_sigmoid = optim.Adam(net_sigmoid.parameters(), lr=lr, weight_decay=5e-4)


scheduler_relu = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_relu, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_lrelu = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_lrelu, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_elu = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_elu, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_gelu = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_gelu, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)
scheduler_sigmoid = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_sigmoid, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)


# Training
def train(epoch):
    net_relu.train()
    net_lrelu.train()
    net_elu.train()
    net_gelu.train()
    net_sigmoid.train()

    train_loss_relu = 0
    train_loss_lrelu = 0
    train_loss_elu = 0
    train_loss_gelu = 0
    train_loss_sigmoid = 0

    correct_relu = 0
    correct_lrelu = 0
    correct_elu = 0
    correct_gelu = 0
    correct_sigmoid = 0

    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_relu.zero_grad()
        optimizer_lrelu.zero_grad()
        optimizer_elu.zero_grad()
        optimizer_gelu.zero_grad()
        optimizer_sigmoid.zero_grad()

        outputs_relu = net_relu(inputs)
        loss_relu = criterion(outputs_relu, targets)
        loss_relu.backward()
        optimizer_relu.step()
        train_loss_relu += loss_relu.item()
        _, predicted_relu = outputs_relu.max(1)
        correct_relu += predicted_relu.eq(targets).sum().item()

        outputs_lrelu = net_lrelu(inputs)
        loss_lrelu = criterion(outputs_lrelu, targets)
        loss_lrelu.backward()
        optimizer_lrelu.step()
        train_loss_lrelu += loss_lrelu.item()
        _, predicted_lrelu = outputs_lrelu.max(1)
        correct_lrelu += predicted_lrelu.eq(targets).sum().item()

        outputs_elu = net_elu(inputs)
        loss_elu = criterion(outputs_elu, targets)
        loss_elu.backward()
        optimizer_elu.step()
        train_loss_elu += loss_elu.item()
        _, predicted_elu = outputs_elu.max(1)
        correct_elu += predicted_elu.eq(targets).sum().item()

        outputs_gelu = net_gelu(inputs)
        loss_gelu = criterion(outputs_gelu, targets)
        loss_gelu.backward()
        optimizer_gelu.step()
        train_loss_gelu += loss_gelu.item()
        _, predicted_gelu = outputs_gelu.max(1)
        correct_gelu += predicted_gelu.eq(targets).sum().item()

        outputs_sigmoid = net_sigmoid(inputs)
        loss_sigmoid = criterion(outputs_sigmoid, targets)
        loss_sigmoid.backward()
        optimizer_sigmoid.step()
        train_loss_sigmoid += loss_sigmoid.item()
        _, predicted_sigmoid = outputs_sigmoid.max(1)
        correct_sigmoid += predicted_sigmoid.eq(targets).sum().item()

        total += targets.size(0)

    epoch_loss_relu = train_loss_relu / ((batch_idx + 1) * train_batch_size)
    epoch_acc_relu = 100.0 * correct_relu / total

    epoch_loss_lrelu = train_loss_lrelu / ((batch_idx + 1) * train_batch_size)
    epoch_acc_lrelu = 100.0 * correct_lrelu / total

    epoch_loss_elu = train_loss_elu / ((batch_idx + 1) * train_batch_size)
    epoch_acc_elu = 100.0 * correct_elu / total

    epoch_loss_gelu = train_loss_gelu / ((batch_idx + 1) * train_batch_size)
    epoch_acc_gelu = 100.0 * correct_gelu / total

    epoch_loss_sigmoid = train_loss_sigmoid / ((batch_idx + 1) * train_batch_size)
    epoch_acc_sigmoid = 100.0 * correct_sigmoid / total

    return (
        epoch_loss_relu,
        epoch_acc_relu,
        epoch_loss_lrelu,
        epoch_acc_lrelu,
        epoch_loss_elu,
        epoch_acc_elu,
        epoch_loss_gelu,
        epoch_acc_gelu,
        epoch_loss_sigmoid,
        epoch_acc_sigmoid,
    )


def test(epoch):
    net_relu.eval()
    net_lrelu.eval()
    net_elu.eval()
    net_gelu.eval()
    net_sigmoid.eval()

    test_loss_relu = 0
    test_loss_lrelu = 0
    test_loss_elu = 0
    test_loss_gelu = 0
    test_loss_sigmoid = 0

    correct_relu = 0
    correct_lrelu = 0
    correct_elu = 0
    correct_gelu = 0
    correct_sigmoid = 0

    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs_relu = net_relu(inputs)
            loss_relu = criterion(outputs_relu, targets)
            test_loss_relu += loss_relu.item()
            _, predicted_relu = outputs_relu.max(1)
            correct_relu += predicted_relu.eq(targets).sum().item()

            outputs_lrelu = net_lrelu(inputs)
            loss_lrelu = criterion(outputs_lrelu, targets)
            test_loss_lrelu += loss_lrelu.item()
            _, predicted_lrelu = outputs_lrelu.max(1)
            correct_lrelu += predicted_lrelu.eq(targets).sum().item()

            outputs_elu = net_elu(inputs)
            loss_elu = criterion(outputs_elu, targets)
            test_loss_elu += loss_elu.item()
            _, predicted_elu = outputs_elu.max(1)
            correct_elu += predicted_elu.eq(targets).sum().item()

            outputs_gelu = net_gelu(inputs)
            loss_gelu = criterion(outputs_gelu, targets)
            test_loss_gelu += loss_gelu.item()
            _, predicted_gelu = outputs_gelu.max(1)
            correct_gelu += predicted_gelu.eq(targets).sum().item()

            outputs_sigmoid = net_sigmoid(inputs)
            loss_sigmoid = criterion(outputs_sigmoid, targets)
            test_loss_sigmoid += loss_sigmoid.item()
            _, predicted_sigmoid = outputs_sigmoid.max(1)
            correct_sigmoid += predicted_sigmoid.eq(targets).sum().item()

            total += targets.size(0)

    epoch_loss_relu = test_loss_relu / ((batch_idx + 1) * test_batch_size)
    epoch_acc_relu = 100.0 * correct_relu / total

    epoch_loss_lrelu = test_loss_lrelu / ((batch_idx + 1) * test_batch_size)
    epoch_acc_lrelu = 100.0 * correct_lrelu / total

    epoch_loss_elu = test_loss_elu / ((batch_idx + 1) * test_batch_size)
    epoch_acc_elu = 100.0 * correct_elu / total

    epoch_loss_gelu = test_loss_gelu / ((batch_idx + 1) * test_batch_size)
    epoch_acc_gelu = 100.0 * correct_gelu / total

    epoch_loss_sigmoid = test_loss_sigmoid / ((batch_idx + 1) * test_batch_size)
    epoch_acc_sigmoid = 100.0 * correct_sigmoid / total

    return (
        epoch_loss_relu,
        epoch_acc_relu,
        epoch_loss_lrelu,
        epoch_acc_lrelu,
        epoch_loss_elu,
        epoch_acc_elu,
        epoch_loss_gelu,
        epoch_acc_gelu,
        epoch_loss_sigmoid,
        epoch_acc_sigmoid,
    )


# Training
train_losses_relu = []
train_losses_lrelu = []
train_losses_elu = []
train_losses_gelu = []
train_losses_sigmoid = []

train_accuracies_relu = []
train_accuracies_lrelu = []
train_accuracies_elu = []
train_accuracies_gelu = []
train_accuracies_sigmoid = []

test_losses_relu = []
test_losses_lrelu = []
test_losses_elu = []
test_losses_gelu = []
test_losses_sigmoid = []

test_accuracies_relu = []
test_accuracies_lrelu = []
test_accuracies_elu = []
test_accuracies_gelu = []
test_accuracies_sigmoid = []

for epoch in range(args.max_epoch):
    print(f"Epoch: {epoch + 1}")

    (
        train_loss_relu,
        train_acc_relu,
        train_loss_lrelu,
        train_acc_lrelu,
        train_loss_elu,
        train_acc_elu,
        train_loss_gelu,
        train_acc_gelu,
        train_loss_sigmoid,
        train_acc_sigmoid,
    ) = train(epoch)

    (
        test_loss_relu,
        test_acc_relu,
        test_loss_lrelu,
        test_acc_lrelu,
        test_loss_elu,
        test_acc_elu,
        test_loss_gelu,
        test_acc_gelu,
        test_loss_sigmoid,
        test_acc_sigmoid,
    ) = test(epoch)

    scheduler_relu.step(test_loss_relu)
    scheduler_lrelu.step(test_loss_lrelu)
    scheduler_elu.step(test_loss_elu)
    scheduler_gelu.step(test_loss_gelu)
    scheduler_sigmoid.step(test_loss_sigmoid)

    # Logging
    train_losses_relu.append(train_loss_relu)
    train_losses_lrelu.append(train_loss_lrelu)
    train_losses_elu.append(train_loss_elu)
    train_losses_gelu.append(train_loss_gelu)
    train_losses_sigmoid.append(train_loss_sigmoid)

    train_accuracies_relu.append(train_acc_relu)
    train_accuracies_lrelu.append(train_acc_lrelu)
    train_accuracies_elu.append(train_acc_elu)
    train_accuracies_gelu.append(train_acc_gelu)
    train_accuracies_sigmoid.append(train_acc_sigmoid)

    test_losses_relu.append(test_loss_relu)
    test_losses_lrelu.append(test_loss_lrelu)
    test_losses_elu.append(test_loss_elu)
    test_losses_gelu.append(test_loss_gelu)
    test_losses_sigmoid.append(test_loss_sigmoid)

    test_accuracies_relu.append(test_acc_relu)
    test_accuracies_lrelu.append(test_acc_lrelu)
    test_accuracies_elu.append(test_acc_elu)
    test_accuracies_gelu.append(test_acc_gelu)
    test_accuracies_sigmoid.append(test_acc_sigmoid)


# Plot
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses_relu, label="ReLU")
plt.plot(train_losses_lrelu, label="Leaky ReLU")
plt.plot(train_losses_elu, label="ELU")
plt.plot(train_losses_gelu, label="GELU")
plt.plot(train_losses_sigmoid, label="Sigmoid")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(train_accuracies_relu, label="ReLU")
plt.plot(train_accuracies_lrelu, label="Leaky ReLU")
plt.plot(train_accuracies_elu, label="ELU")
plt.plot(train_accuracies_gelu, label="GELU")
plt.plot(train_accuracies_sigmoid, label="Sigmoid")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(test_losses_relu, label="ReLU")
plt.plot(test_losses_lrelu, label="Leaky ReLU")
plt.plot(test_losses_elu, label="ELU")
plt.plot(test_losses_gelu, label="GELU")
plt.plot(test_losses_sigmoid, label="Sigmoid")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(test_accuracies_relu, label="ReLU")
plt.plot(test_accuracies_lrelu, label="Leaky ReLU")
plt.plot(test_accuracies_elu, label="ELU")
plt.plot(test_accuracies_gelu, label="GELU")
plt.plot(test_accuracies_sigmoid, label="Sigmoid")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

if not os.path.isdir('log'):
    os.mkdir('log')
plt.savefig(f'./log/activation.png')

print("Done")
