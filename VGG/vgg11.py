PATH = '/content/drive/MyDrive/vgg_c10_aa_ls.pth'

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration for VGG
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# VGG Model Definition
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# Load Pre-trained Weights
state_dict = torch.load(PATH)
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '')] = state_dict.pop(key)

model = VGG('VGG11')
model.load_state_dict(state_dict)
model.eval()

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 128
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyperparameter Tuning Section (Example - Dummy Grid)
learning_rates = [0.001, 0.01]
best_lr = None
best_accuracy = 0

for lr in learning_rates:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Dummy training loop (placeholder)
    # Replace with your actual training logic
    print(f"Evaluating learning rate: {lr}")
    # Assume the current lr gives certain accuracy; track the best
    current_accuracy = 85.0  # Example placeholder accuracy
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_lr = lr

print(f"Best learning rate: {best_lr} with accuracy: {best_accuracy}%")

# Model Evaluation Metrics
model.cuda()
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Class-wise Accuracy
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

for label, prediction in zip(all_labels, all_predictions):
    if label == prediction:
        correct_pred[classes[label]] += 1
    total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# Overall Accuracy
total_correct = sum(correct_pred.values())
total_samples = sum(total_pred.values())
overall_accuracy = 100 * total_correct / total_samples
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
