import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.tensorboard import SummaryWriter


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = f.relu(out)
        return out

class FlowersClassifier(nn.Module):
    def __init__(self, num_classes, num_units=64):
        super(FlowersClassifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, num_units, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlockSE(num_units, num_units),
            ResidualBlockSE(num_units, num_units),
            ResidualBlockSE(num_units, num_units * 2, stride=2),
            ResidualBlockSE(num_units * 2, num_units * 2),
            ResidualBlockSE(num_units * 2, num_units * 4, stride=2),
            ResidualBlockSE(num_units * 4, num_units * 4),
            ResidualBlockSE(num_units * 4, num_units * 8, stride=2),
            ResidualBlockSE(num_units * 8, num_units * 8),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_units * 8, num_classes)
        )


    def forward(self, x):
        return self.layers(x)


# Training function
def train(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log to TensorBoard
        writer.add_scalar('Training/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Training/Accuracy', 100 * correct / total, epoch * len(train_loader) + batch_idx)

        if (batch_idx + 1) % 5 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(train_loader)}] - "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {100 * correct / total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    # Log epoch metrics
    writer.add_scalar('Training/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Training/Epoch_Accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_acc



# Evaluation function
def evaluate(model, test_loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log to TensorBoard
            writer.add_scalar('Validation/Loss', loss.item(), epoch * len(test_loader) + batch_idx)
            writer.add_scalar('Validation/Accuracy', 100 * correct / total, epoch * len(test_loader) + batch_idx)

            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f"    Batch [{batch_idx + 1}/{len(test_loader)}] - "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {100 * correct / total:.2f}%")

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total

    # Log epoch metrics
    writer.add_scalar('Validation/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Validation/Epoch_Accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_acc





