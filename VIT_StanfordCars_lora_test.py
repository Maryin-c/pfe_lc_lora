import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import StanfordCars
from torch.utils.data import DataLoader, Subset
import numpy as np

from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor
from torch import nn, optim
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model, TaskType

import wandb

batch_size = 128
num_classes = 196
num_epochs = 10
learning_rate = 5e-5
model_path = "google/vit-base-patch16-224-in21k"

task_type = TaskType.IMAGE_CLASSIFICATION
inference_mode = False
rank = 8
lora_alpha = 16
bias = "none"
lora_dropout = 0.1
target_modules = ["query", "key", "value", "intermediate.dense", "output.dense"]

wandb.login(key="d8a57853232ad9c5337ec726db40457ebbf81f1a")
run = wandb.init(
    # Set the wandb project where this run will be logged.
    project="vit-stanford-cars-incremental-test",
    # Track hyperparameters and run metadata.
    config={
        "model_path": model_path,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_classes": num_classes,
    },
)

processor = ViTImageProcessor.from_pretrained(model_path)

# 任务数据集划分
def get_task_data(dataset, task_classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in task_classes]
    return Subset(dataset, indices)

def data_loader():
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    print("mean: {} , std: {}".format(feature_extractor.image_mean, feature_extractor.image_std), flush=True)
    # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 加载数据集
    train_data = StanfordCars(root="./data", split="train", download=False, transform=train_transform)
    test_data = StanfordCars(root="./data", split="test", download=False, transform=test_transform)

    tasks = [list(range(i, min(i + 50, num_classes))) for i in range(0, num_classes, 50)]

    train_loaders = []
    test_loaders = []
    for i in range(len(tasks)):
        train_task = get_task_data(train_data, tasks[i])
        test_task = get_task_data(test_data, tasks[i])
        train_loader = DataLoader(train_task, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


train_loaders, test_loaders = data_loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(model_path, num_labels=num_classes)

# 配置 LoRA 层
lora_config = LoraConfig(
    task_type=task_type,
    inference_mode=inference_mode,
    r=rank,  # 低秩矩阵的秩
    lora_alpha=lora_alpha,  # LoRA 影响因子
    target_modules=target_modules,  # 仅作用于 Self-Attention 层
    lora_dropout=lora_dropout,  # Dropout 防止过拟合
    bias=bias
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 检查 LoRA 训练参数
model.print_trainable_parameters()
model = model.to(device)

def evaluate(model, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs).logits
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    test_acc = 100. * correct / total
    return test_acc

# 训练函数
def train_model(model, train_loader, test_loader, num_epochs, id):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            inputs = processor(images=images, return_tensors="pt")
            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            run.log({"loss": loss.item()})

        train_acc = 100. * correct / total

        model.eval()
        
        test_acc = evaluate(model, test_loader)

        total_correct = 0
        total_samples = 0
        for i in range(id + 1):
            correct = 0
            samples = 0
            for images, labels in test_loaders[i]:
                images, labels = images.to(device), labels.to(device)
                inputs = processor(images=images, return_tensors="pt")
                outputs = model(**inputs).logits
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                samples += labels.size(0)

            total_correct += correct
            total_samples += samples
        total_test_acc = 100. * total_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Total Test Acc: {total_test_acc:.2f}%", flush=True)
        run.log({
            "Train Loss": running_loss/len(train_loader), 
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Total Test Acc": total_test_acc,
            })

    return model

for i in range(len(train_loaders)):
    model = model.to(device)
    model = train_model(model, train_loaders[i], test_loaders[i], num_epochs, i)

    # model.merge_and_unload()
    # model = get_peft_model(model, lora_config)

    model = model.to('cpu')
    # torch.save(model.state_dict(), "vit_task_lora_{}.pth".format(i))
    model.save_pretrained("vit_task_lora_{}.pth".format(i))

    
run.finish()
