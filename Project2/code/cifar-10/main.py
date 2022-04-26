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
parser.add_argument(
    '--resume', '-r', action='store_true', help="Resume from checkpoint"
)
parser.add_argument(
    '--model',
    default='ResNet18',
    type=str,
    help="ResNet18, ResNet50, PreActResNet18, ResNeXt29_32x4d, ResNeXt29_2x64d, \
        WideResNet28x10, DenseNet121, DPN26, DLA",
)
parser.add_argument('--lr', default=0.1, type=float, help="Learning rate")
parser.add_argument('--max-epoch', default=200, type=int, help="Max training epochs")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # Best test accuracy
start_epoch = 0  # Start from epoch 0 or last checkpoint epoch

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
print(f"==> Using model {args.model}")
net = getattr(models, args.model)().to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint...")
    assert os.path.isdir('checkpoints'), "Error: no checkpoints directory found!"
    checkpoint = torch.load(f'./checkpoints/checkpoint_{args.model}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = train_loss / ((batch_idx + 1) * train_batch_size)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = test_loss / ((batch_idx + 1) * test_batch_size)
    epoch_acc = 100.0 * correct / total

    # Save checkpoint
    if epoch_acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/checkpoint_{args.model.lower()}.pth')
        best_acc = epoch_acc

    return epoch_loss, epoch_acc


# Training
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
lr_schedule = []
for epoch in range(start_epoch, start_epoch + args.max_epoch):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step(test_loss)

    # Print log info
    print("============================================================")
    print(f"Epoch: {epoch + 1}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%"
    )
    print("============================================================")

    # Logging
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    lr_schedule.append(optimizer.param_groups[0]['lr'])


# Plot
plt.figure(figsize=(16, 8))

plt.subplot2grid((2, 4), (0, 0), colspan=3)
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot2grid((2, 4), (1, 0), colspan=3)
plt.plot(train_accuracies, label='train')
plt.plot(test_accuracies, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot2grid((2, 4), (0, 3), rowspan=2)
plt.plot(lr_schedule, label='lr')
plt.legend()
plt.title('Learning Rate')
plt.xlabel('Epoch')

if not os.path.isdir('log'):
    os.mkdir('log')
plt.savefig(f'./log/{args.model.lower()}.png')
