
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64


transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = trainset.classes


def naive_predict(labels):
    return [random.randint(0, 9) for _ in labels]


all_labels = [label for _, label in testset]
naive_preds = naive_predict(all_labels)
naive_acc = sum([p==t for p,t in zip(naive_preds, all_labels)])/len(all_labels)
print(f"Naive Baseline Accuracy: {naive_acc:.4f}")


resize_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
resnet18.eval()
resnet18.to(device)


correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in testloader:
        imgs = torch.stack([resize_transform(transforms.ToPILImage()(img)) for img in imgs]).to(device)
        labels = labels.to(device)
        outputs = resnet18(imgs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Pretrained ResNet18 Accuracy (no fine-tune): {correct/total:.4f}")
