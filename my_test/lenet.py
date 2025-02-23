#!/usr/bin/env python
# coding: utf-8

# In[20]:


TASK_NAME = "lenet_mnist"


# In[21]:


import sys
import os

# 获取当前 notebook 的路径
notebook_path = os.getcwd()
parent_path = os.path.dirname(notebook_path)  # 获取上一级目录路径

print(notebook_path)
print(parent_path)

# 添加到 sys.path
#sys.path.append(parent_path)


# In[22]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from src.models.LeNet import LeNet
import numpy as np

from src.models.LeNet_LowRank import getBase, LeNet_LowRank, load_sd_decomp
import src.main as lc
from src.utils.utils import evaluate_accuracy, lazy_restore, evaluate_compression
import old_lc.main as olc

import matplotlib.pyplot as plt

import math
import json

# In[23]:


# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 选择设备（GPU or CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[24]:


PRE_TRAINED = "./my_test/" + TASK_NAME + "/pretrained_model.pth"
if not os.path.exists("./my_test/" + TASK_NAME):
    os.makedirs("./my_test/" + TASK_NAME)


# ## 创建第一个检查点

# In[25]:


def accuracy_cal(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# In[26]:


# 初始化模型
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
# optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Adam 优化器

accuracy_threshold = 75
# 训练模型
num_epochs = 10

stop_training = False
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for iter, data in enumerate(trainloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
        if iter % 100 == 0:
            accuracy = accuracy_cal(model) 
            model.train()
            print(f"Epoch {epoch+1}, Accuracy {accuracy:.2f}%, Loss: {running_loss / len(trainloader):.4f}")
            if accuracy >= accuracy_threshold:
                stop_training = True
                print(f"Accuracy reached {accuracy_threshold}%, saving model and stopping training.")
                torch.save(model.state_dict(), PRE_TRAINED)  # 保存模型
                break
    if stop_training:
        break


# ## 使用delta-lora、lc-checkpoint、lc+delta与baseline对比

# In[27]:


def train_model(model, optimizer, images, labels):
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # print("LoRA+LC Training Loss (Decomposed): {}".format(loss.item()))


# In[28]:


DECOMPOSED_LAYERS = ['classifier.1.weight', 'classifier.3.weight']
RANK = -1
SCALING = -1

DLORA_LC_LOC = "./my_test/" + TASK_NAME + "/dlora-lc"
if not os.path.exists(DLORA_LC_LOC):
    os.makedirs(DLORA_LC_LOC)

LC_LOC = "./my_test/" + TASK_NAME + "/lc"
if not os.path.exists(LC_LOC):
    os.makedirs(LC_LOC)


# In[29]:


# 训练模型
num_epochs = 20

full_accuracy = []
decomposed_full_accuracy = []
restored_accuracy = []
lc_accuracy = []

# 不使用lora

baseline_model = LeNet().to(device)
lc_checkpoint_model = LeNet().to(device)

baseline_model.load_state_dict(torch.load(PRE_TRAINED))
lc_checkpoint_model.load_state_dict(torch.load(PRE_TRAINED))

# 使用lora

w, b = getBase(baseline_model)
delta_lora_model = LeNet_LowRank(w, b, rank = RANK).to(device)
dlora_lc_model = LeNet_LowRank(w, b, rank = RANK).to(device)

load_sd_decomp(torch.load(PRE_TRAINED), delta_lora_model, DECOMPOSED_LAYERS)
load_sd_decomp(torch.load(PRE_TRAINED), dlora_lc_model, DECOMPOSED_LAYERS)

# 对应的优化器
learning_rate = 0.01
baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr = learning_rate)
lc_checkpoint_optimizer = torch.optim.SGD(lc_checkpoint_model.parameters(), lr = learning_rate)
delta_lora_optimizer = torch.optim.SGD(delta_lora_model.parameters(), lr = learning_rate)
dlora_lc_optimizer = torch.optim.SGD(dlora_lc_model.parameters(), lr = learning_rate)

# delta-lc压缩，创建第一个压缩点
current_set = 0
current_iter = 0
set_path = "/set_{}".format(current_set)
if not os.path.exists(DLORA_LC_LOC + set_path):
    os.makedirs(DLORA_LC_LOC + set_path)
dlora_lc_model = dlora_lc_model.to('cpu')
dlora_lc_weights, dlora_lc_decomp_weights = lc.extract_weights(dlora_lc_model, DLORA_LC_LOC + "/set_{}".format(current_set), DECOMPOSED_LAYERS)
dlora_lc_model = dlora_lc_model.to(device)
# 上一个基线检查点，用于模拟恢复
last_dlora_lc_baseline_checkpoint = LeNet()
last_dlora_lc_baseline_checkpoint.load_state_dict(torch.load(PRE_TRAINED))


# lc 压缩
current_iter_old_lc = 0
current_set_old_lc = 0

lc_checkpoint_model = lc_checkpoint_model.to('cpu')
cstate = lc_checkpoint_model.state_dict()
set_path = "/set_{}".format(current_set_old_lc)
if not os.path.exists(LC_LOC + set_path):
    os.makedirs(LC_LOC + set_path)
prev_state = olc.extract_weights(cstate, LC_LOC + set_path, DECOMPOSED_LAYERS)
lc_checkpoint_model = lc_checkpoint_model.to(device)

# 训练
for epoch in range(num_epochs):
    for iter, data in enumerate(trainloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
            
        # baseline训练
        train_model(baseline_model, baseline_optimizer, images, labels)

        # lc-checkpoint训练
        train_model(lc_checkpoint_model, lc_checkpoint_optimizer, images, labels)

        # delta-lora训练
        # train_model(delta_lora_model, delta_lora_optimizer, images, labels)

        # delta-lc训练
        train_model(dlora_lc_model, dlora_lc_optimizer, images, labels)
    
        ########################################
        # dlora-lc部分
        ########################################
        # 每10个iteration保存新的基线检查点
        if iter == 0 and epoch == 0:
            pass
        elif iter % 10 == 0:
            # 模拟恢复：
            # 1.读取最近的基线检查点，这里持久保存最近的基线检查点，因此可以减少一个读取过程
            # 2.基线检查点加一堆delta lora来恢复到最新的完整检查点
            last_dlora_lc_baseline_checkpoint = lazy_restore(dlora_lc_weights, dlora_lc_decomp_weights, bias, LeNet(), 
                                                    last_dlora_lc_baseline_checkpoint.state_dict(), DECOMPOSED_LAYERS, rank = RANK, scaling = SCALING)
            
            current_set += 1
            current_iter = 0

            # 3.保存最新的检查点
            set_path = "/set_{}".format(current_set)
            if not os.path.exists(DLORA_LC_LOC + set_path):
                os.makedirs(DLORA_LC_LOC + set_path)
            
            # Rebuilding LoRA layers => reset model!
            w, b = getBase(last_dlora_lc_baseline_checkpoint)
            dlora_lc_model = LeNet_LowRank(w, b, rank = RANK)
            dlora_lc_optimizer = torch.optim.SGD(dlora_lc_model.parameters(), lr = learning_rate)
            load_sd_decomp(last_dlora_lc_baseline_checkpoint.state_dict(), dlora_lc_model, DECOMPOSED_LAYERS)
            
            dlora_lc_model = dlora_lc_model.to('cpu')
            dlora_lc_weights, dlora_lc_decomp_weights = lc.extract_weights(dlora_lc_model, DLORA_LC_LOC + 
                                                    "/set_{}".format(current_set), DECOMPOSED_LAYERS)
            dlora_lc_model = dlora_lc_model.to(device)
        else:
            # delta lora
            dlora_lc_model = dlora_lc_model.to('cpu')
            delta, decomp_delta, bias = lc.generate_delta(dlora_lc_weights, dlora_lc_decomp_weights, dlora_lc_model.state_dict(), DECOMPOSED_LAYERS)
            dlora_lc_model = dlora_lc_model.to(device)
            # lc checkpoint compression
            compressed_delta, full_delta, compressed_dcomp_delta, full_dcomp_delta  = lc.compress_delta(delta, decomp_delta)
            # save
            lc.save_checkpoint(compressed_delta, compressed_dcomp_delta, bias, current_iter, DLORA_LC_LOC + "/set_{}".format(current_set))
            # update weights
            dlora_lc_weights = np.add(dlora_lc_weights, full_delta) # Replace base with latest for delta to accumulate.
            dlora_lc_decomp_weights = np.add(dlora_lc_decomp_weights, full_dcomp_delta)

            current_iter += 1

        ########################################
        # lc部分
        ########################################
        if iter == 0 and epoch == 0:
            pass
        else:
            if iter % 10 == 0:
                lc_checkpoint_model = lc_checkpoint_model.to('cpu')
                cstate = lc_checkpoint_model.state_dict()
                current_set_old_lc += 1
                current_iter_old_lc = 0
                set_path = "/set_{}".format(current_set_old_lc)
                if not os.path.exists(LC_LOC + set_path):
                    os.makedirs(LC_LOC + set_path)
                # torch.save(cstate, SAVE_LOC_OLC + set_path + "/initial_model.pt")
                prev_state = olc.extract_weights(cstate, LC_LOC + set_path, DECOMPOSED_LAYERS)
                lc_checkpoint_model = lc_checkpoint_model.to(device)
            else:
                lc_checkpoint_model = lc_checkpoint_model.to('cpu')
                cstate = lc_checkpoint_model.state_dict()
                old_lc_delta, old_lc_bias = olc.generate_delta(prev_state, cstate, DECOMPOSED_LAYERS)
                # print("Compressing delta for old_lc")
                # compressed_delta = olc.compress_delta(old_lc_delta, num_bits = 3)
                olc_compressed_delta, update_prev = olc.compress_data(old_lc_delta, num_bits = 3)
                olc.save_checkpoint(LC_LOC + "/set_{}".format(current_set_old_lc), olc_compressed_delta, 
                                    old_lc_bias, current_iter_old_lc)
                prev_state = np.add(prev_state, update_prev)
                current_iter_old_lc += 1
                lc_checkpoint_model = lc_checkpoint_model.to(device)

        # if iter % 100 == 0:
        #     accuracy = accuracy_cal(dlora_lc_model) 
        #     dlora_lc_model.train()

        #     baseline_acc = accuracy_cal(baseline_model)
        #     baseline_model.train()

        #     restored_model = lazy_restore(dlora_lc_weights, base_decomp, bias, LeNet(), 
        #                                   original.state_dict(), DECOMPOSED_LAYERS, 
        #                                   rank = RANK, scaling = SCALING)
        #     restored_lc_model = LeNet().to(device)
        #     restored_lc_model.load_state_dict(olc.restore_state_dict(prev_state, old_lc_bias, 
        #                                                           restored_model.state_dict(), DECOMPOSED_LAYERS))
            
        #     lc_accuracy = accuracy_cal(lc_checkpoint_model)

        #     print(f"Epoch {epoch+1}, DLORA LC Accuracy {accuracy:.2f}%, baseline acc {baseline_acc:.2f}%")
    
        # if iter % 100 == 0 and iter != 0:
        #     full_accuracy = accuracy_cal(baseline_model)
        #     decomposed_full_accuracy = accuracy_cal(dlora_lc_model)
        #     restored_model = lazy_restore(dlora_lc_weights, dlora_lc_decomp_weights, bias, LeNet(), 
        #                                   last_dlora_lc_baseline_checkpoint.state_dict(), DECOMPOSED_LAYERS, 
        #                                   rank = RANK, scaling = SCALING)
        #     restored_model = restored_model.to(device)
        #     restored_accuracy = accuracy_cal(restored_model)
        #     restored_model = restored_model.to('cpu')
        #     restored_lc_model = LeNet()
        #     restored_lc_model.load_state_dict(olc.restore_state_dict(prev_state, old_lc_bias, 
        #                                                           restored_model.state_dict(), DECOMPOSED_LAYERS))
        #     restored_lc_model = restored_lc_model.to(device)
        #     lc_accuracy = accuracy_cal(restored_lc_model)
        #     restored_lc_model = restored_lc_model.to('cpu')
        #     print("Full accuracy: {}, LC accuracy: {}, Decomposed-Full accuracy: {}, Decomposed-Restored accuracy: {}".format(
        #         full_accuracy, lc_accuracy, decomposed_full_accuracy, restored_accuracy))

        if iter % 100 == 0 and iter != 0:
            full_accuracy.append(accuracy_cal(baseline_model))
            decomposed_full_accuracy.append(accuracy_cal(dlora_lc_model))
            restored_model = lazy_restore(dlora_lc_weights, dlora_lc_decomp_weights, bias, LeNet(), 
                                          last_dlora_lc_baseline_checkpoint.state_dict(), DECOMPOSED_LAYERS, 
                                          rank = RANK, scaling = SCALING)
            
            restored_model = restored_model.to(device)
            restored_accuracy.append(accuracy_cal(restored_model))
            restored_model = restored_model.to('cpu')
            restored_lc_model = LeNet()
            restored_lc_model.load_state_dict(olc.restore_state_dict(prev_state, old_lc_bias, 
                                                                  restored_model.state_dict(), DECOMPOSED_LAYERS))
            restored_lc_model = restored_lc_model.to(device)
            lc_accuracy.append(accuracy_cal(restored_lc_model))
            restored_lc_model = restored_lc_model.to('cpu')
            print("Full accuracy: {}, LC accuracy: {}, Decomposed-Full accuracy: {}, Decomposed-Restored accuracy: {}".format(
                full_accuracy[-1], lc_accuracy[-1], decomposed_full_accuracy[-1], restored_accuracy[-1]))


# ## 画图

# In[36]:



plt.figure(figsize = (30, 5))
plt.title("LeNet, Accuracy")
plt.plot(full_accuracy, label = "Default LeNet")
plt.plot(lc_accuracy, label = "LC LeNet")
plt.plot(decomposed_full_accuracy, label = "dLoRA LeNet")
plt.plot(restored_accuracy, label = "dLoRA + LC LeNet")
plt.xticks([x for x in range(0, 120) if x % 6 == 0], [x for x in range(0, 20)])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
# plt.show()

plt.savefig("./my_test/" + TASK_NAME + "/lenet_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()  # 关闭图像，释放资源


# In[37]:


rangex = [x for x in range(0, 120) if x % 6 == 0]
rangey = [x for x in range(0, 20)]
plt.figure(figsize = (40, 10))
ax1 = plt.subplot(1, 2, 1)
ax1.set_title("LeNet Absolute Accuracy Loss (Default LeNet vs LC + dLoRA LeNet / LC LeNet)")
plt.plot(np.abs(np.subtract(np.array(full_accuracy), 
                     np.array(restored_accuracy))), label = "LC + dLoRA LeNet")
plt.plot(np.abs(np.subtract(np.array(full_accuracy), 
                     np.array(lc_accuracy))), label = "LC LeNet")
plt.legend()
plt.xticks(rangex, rangey)
plt.ylabel("Absolute Accuracy Loss")
plt.xlabel("Epoch")
plt.axhline(y = 0.05, color = 'r')
plt.ylim(0, 0.5)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title("LeNet Absolute Restoration Accuracy Loss (LC + dLoRA LeNet & LC LeNet)")
plt.plot(np.abs(np.subtract(np.array(restored_accuracy), 
                     np.array(decomposed_full_accuracy))), label = "LC + dLoRA LeNet")
plt.plot(np.abs(np.subtract(np.array(full_accuracy), 
                     np.array(lc_accuracy))), label = "LC LeNet")
plt.legend()
plt.ylim(0, 0.5)
plt.axhline(y = 0.05, color = 'r')
plt.xticks(rangex, rangey)
plt.ylabel("Absolute Accuracy Loss")
plt.xlabel("Epoch")
# plt.show()

plt.savefig("./my_test/" + TASK_NAME + "/lenet_loss.png", dpi=300, bbox_inches='tight')
plt.close()  # 关闭图像，释放资源


# In[32]:



def getsize(sl):
    dir = [x for x in os.listdir(sl)]
    csize, usize = 0, 0
    for set in dir:
        for f in os.listdir(sl + "/" + set):
            fp = sl + "/{}/{}".format(set, f)
            csize += os.path.getsize(fp)
            usize += 250 * math.pow(2, 10) # torch checkpoint same size
    return csize, usize,


# In[33]:


compressed_size, uncompressed_size = getsize(DLORA_LC_LOC)
a, b = evaluate_compression(uncompressed_size, compressed_size)
compressed_size, uncompressed_size = getsize(LC_LOC)
a1, b1 = evaluate_compression(uncompressed_size, compressed_size)

print("LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a1, b1))
print("LoRA + LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a, b))


# In[34]:



data = {
    "full_acc" : full_accuracy,
    "decomposed_restored_accuracy" : restored_accuracy,
    "decomposed_full_accuracy" : decomposed_full_accuracy,
    "lc_restored_accuracy" : lc_accuracy
}
with open("./my_test/" + TASK_NAME + "/data.json", 'w') as f:
    json.dump(data, f)

