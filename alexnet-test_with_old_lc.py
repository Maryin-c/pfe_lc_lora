#!/usr/bin/env python
# coding: utf-8

# # AlexNet Implementation with Old_LC on MNIST 

# In[2]:


import glob
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy as spy
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import ssl
import pickle, json
import src.main as lc
import old_lc.main as olc
from src.models.AlexNet import AlexNet
import torchvision
import src.compression.deltaCompress as lc_compress
from src.models.AlexNet_LowRank import getBase, AlexNet_LowRank, load_sd_decomp
from src.utils.utils import evaluate_accuracy, lazy_restore, evaluate_compression


# In[3]:


# Function to count the number of parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    
    # Exclude the last nn.Linear layer
    linear_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != 'classifier.6':
            linear_params += sum(p.numel() for p in module.parameters())

    return total_params, linear_params

model = AlexNet()
# Get the total and linear parameters
total_params, linear_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Linear parameters (excluding the last nn.Linear): {linear_params}")


# ## Definition of Data Loader function

# In[2]:


# HDFP = "./volumes/Ultra Touch" # Load HHD
HDFP = "." # Load HHD

def data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    trainset = datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    # Reintroduce the 2000 datapoints model has not seen before.
    # trainset.data = trainset.data.clone()[-2000:-1000]
    # trainset.targets = trainset.targets.clone()[-2000:-1000]
    trainset.data = trainset.data.clone()[-2000:-1000]
    trainset.targets = trainset.targets.clone()[-2000:-1000]
    # trainset.data = trainset.data.clone()[-51000:]
    # trainset.targets = trainset.targets.clone()[-51000:]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32,
                                              shuffle=False, num_workers=2)

    testset = datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    testset.data = trainset.data[-1000:]
    testset.targets = trainset.targets[-1000:]
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader


# HDFP = "./volumes/Ultra Touch"  # Placeholder for HDD path

# def data_loader():
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     # Load the MNIST training dataset
#     trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     # Use the last 10,000 images for training
#     trainset.data = trainset.data.clone()[-56000:]
#     trainset.targets = trainset.targets.clone()[-56000:]
#     # trainset.data = trainset.data.clone()[-58000:]
#     # trainset.targets = trainset.targets.clone()[-58000:]
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

#     # Load the MNIST test dataset
#     testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     # Use the first 1,000 images from the test dataset
#     testset.data = testset.data.clone()[:1000]
#     testset.targets = testset.targets.clone()[:1000]
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
#     return train_loader, test_loader


# ## Calling MNIST dataset

# In[3]:


# Bypass using SSL unverified
ssl._create_default_https_context = ssl._create_unverified_context
# MNIST dataset 
train_loader, test_loader = data_loader()


# ## Bypass the matplotlib error

# In[4]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# ## Showing some images of the dataset we use

# In[5]:


# Adjust these values to match the normalization values used during the loading of your dataset
mean = 0.1307
std = 0.3081

# Function to show an image
def imshow(img):
    # Adjusting unnormalization for potentially 3-channel images
    img = img * std + mean  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Assuming train_loader is defined and loaded as before
dataiter = iter(train_loader)
images, labels = next(dataiter)

# # Show images
# imshow(torchvision.utils.make_grid(images[:3]))
# Print labels
# print(' '.join('%5s' % labels[j] for j in range(3)))


# ## Creation of folder to save the results (for plots and compression rate)

# In[6]:


SAVE_LOC = HDFP + "/alexnet-lobranch"
if not os.path.exists(SAVE_LOC):
    os.makedirs(SAVE_LOC)

SAVE_LOC_OLC = HDFP + "/alexnet-old-lc"
if not os.path.exists(SAVE_LOC_OLC):
    os.makedirs(SAVE_LOC_OLC)


# ## Definition of the accuracy functions

# In[7]:


def accuracy_binary(model, evaluation_set):
    model.eval()  # Switches the model to evaluation mode.

    no_correct, no_seen = 0, 0  # Initialize counters for correct predictions and total samples seen.

    with torch.no_grad():  # Disables gradient calculation.
        for input, label in evaluation_set:  # Iterate over the evaluation dataset.
            output = torch.sigmoid(model(input))  # Apply sigmoid to model output to get probabilities.
            output = torch.where(output > 0.5, 1, 0)  # Threshold probabilities at 0.5 to decide between classes 0 and 1.
            no_seen += label.size(0)  # Count the number of samples seen (batch size).
            no_correct += (output == label).sum().item()  # Increment correct predictions by the number of matches in the batch.
    
    acc = no_correct / no_seen  # Calculate accuracy as the ratio of correct predictions to total samples.
    model.train()  # Switch the model back to training mode.
    return acc  # Return the computed accuracy.

def accuracy_multiclass(model, evaluation_set):
    model.eval()  # Switches the model to evaluation mode.

    no_correct, no_seen = 0, 0  # Initialize counters for correct predictions and total samples seen.

    with torch.no_grad():  # Disables gradient calculation.
        for input, label in evaluation_set:  # Iterate over the evaluation dataset.
            output = model(input)  # Get the raw logits from the model.
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max logit which represents the predicted class.
            no_seen += label.size(0)  # Count the number of samples seen (batch size).
            no_correct += pred.eq(label.view_as(pred)).sum().item()  # Compare predictions with true labels and sum up correct predictions.
    
    acc = no_correct / no_seen  # Calculate accuracy as the ratio of correct predictions to total samples.
    model.train()  # Switch the model back to training mode.
    return acc  # Return the computed accuracy.


# ## Special function for the accuracy of the model on GPU 

# In[8]:


def accuracy_multiclass_gpu(model, evaluation_set):
    device = next(model.parameters()).device  # Get the device of the model
    model.eval()  # Switches the model to evaluation mode.

    no_correct, no_seen = 0, 0  # Initialize counters for correct predictions and total samples seen.

    with torch.no_grad():  # Disables gradient calculation.
        for inputs, labels in evaluation_set:  # Iterate over the evaluation dataset.
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device of the model
            output = model(inputs)  # Get the raw logits from the model.
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max logit which represents the predicted class.
            no_seen += labels.size(0)  # Count the number of samples seen (batch size).
            no_correct += pred.eq(labels.view_as(pred)).sum().item()  # Compare predictions with true labels and sum up correct predictions.
    
    acc = no_correct / no_seen  # Calculate accuracy as the ratio of correct predictions to total samples.
    model.train()  # Switch the model back to training mode.
    return acc  # Return the computed accuracy.


# ## Usual training on GPU (Creating branchpoints)

# In[ ]:


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# get the base model (VGG16NoLite) and move it to the chosen device
model_for_checkpoint = AlexNet().to(device)

# creating branchpoints
epochs = 20
isLoop = True
optimizer = torch.optim.SGD(model_for_checkpoint.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4) 
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for iter, data in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)

        # optimizer.zero_grad()

        # output = model_for_checkpoint(data)

        # loss = criterion(output, target)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device

        optimizer.zero_grad()
        outputs = model_for_checkpoint(inputs)

        # Here assuming your loss function and any other operation are compatible with CUDA tensors
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()



        if iter % 20 == 0:
            # print("Running validation for {} Epoch, {} Iteration...".format(epoch, iter))
            res = accuracy_multiclass_gpu(model_for_checkpoint, test_loader)  # Ensure this function also handles data on GPU

            # print("ACCURACY: {}".format(res))
            if res > 0:
                # Move model to CPU before saving
                model_for_checkpoint.to('cpu')
                torch.save(model_for_checkpoint.state_dict(), HDFP + "/branch_{}.pt".format(res))
                # Optionally, move model back to the original device (GPU) if further computation is needed
                model_for_checkpoint.to(device)
                print("Model saved at accuracy: ", res)

            if res > 0.96:
                isLoop = False
                # print("Model saved at accuracy: ", res)
                break
        
    print("Epoch : [{}/{}], Training Loss: {}".format(epoch, epochs-1, loss.item()))
    if not isLoop:
        break
    print("Length of train_loader is: ", len(train_loader))


# In[ ]:


# get the base model (AlexNet)

model_for_checkpoint = AlexNet()

# creating branchpoints : 

epochs = 10
isLoop = True
optimizer = torch.optim.SGD(model_for_checkpoint.parameters(), lr=0.01, momentum=0.9) # momentum=0.9

for epoch in range(epochs):
    for iter, data in enumerate(train_loader):
        inputs, labels = data
        # print(inputs, labels)
        optimizer.zero_grad()
        outputs = model_for_checkpoint(inputs)

        # if self.config.loss_function == "binary_cross_entropy":
        #     outputs = torch.sigmoid(outputs)
        
        # loss = loss_function(outputs, labels)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        # print("Epoch {} | Iteration {} : Loss {}".format(epoch, iter, loss.item()))
        if iter % 20 == 0:
            print("Running validation for {} Epoch, {} Iteration...".format(epoch, iter))
            # Previously: res = accuracy_binary(model_for_checkpoint, test_loader)
            res = accuracy_multiclass(model_for_checkpoint, test_loader)

            # res = accuracy_binary(model_for_checkpoint, test_loader)
            
            print("ACCURACY: {}".format(res))
            if res > 0.7:
                print("Model saved at accuracy: ", res)
                # torch.save(model_for_checkpoint.state_dict(), HDFP + "/lobranch-snapshot/branchpoints/alexnet/branch_{}.pt".format(res))
            if res > 0.9:
                isLoop = False
                # break
    if not isLoop:
        break
    print("Length of train_loader is: ", len(train_loader))


# ## Exploiting branchpoints of the model

# ### Call of the different models to compare

# In[9]:


# DECOMPOSED_LAYERS = ["classifier.1.weight", "classifier.4.weight", "classifier.6.weight"]
DECOMPOSED_LAYERS = ["classifier.1.weight", "classifier.4.weight"]

RANK = -1
SCALING = -1
BRANCH_ACC = "0.805"

# Set up weights for original AlexNet model
original = AlexNet()
model_original = AlexNet()

# Load from "branch point"
BRANCH_LOC = HDFP + "/branch_{}.pt".format(BRANCH_ACC)
original.load_state_dict(torch.load(BRANCH_LOC))
model_original.load_state_dict(torch.load(BRANCH_LOC))

w, b = getBase(model_original)
model = AlexNet_LowRank(w, b, rank = RANK)
load_sd_decomp(torch.load(BRANCH_LOC), model, DECOMPOSED_LAYERS)
learning_rate = 0.01
# momentum = 0.9
# weight_decay=5e-4
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer_full = torch.optim.SGD(model_original.parameters(), lr = learning_rate)


# In[10]:


# print model low rank
print(model)


# In[11]:


delta_normal_max = []
delta_normal_min = []
delta_decomposed_max = []
delta_decomposed_min = []
full_accuracy = []
decomposed_full_accuracy = []
restored_accuracy = []
lc_accuracy = []
current_iter = 0
current_set = 0

current_iter_old_lc = 0
current_set_old_lc = 0

acc = lambda x, y : (torch.max(x, 1)[1] == y).sum().item() / y.size(0)

for epch in range(20):
    for i, data in enumerate(train_loader, 0):
        print("Epoch: {}, Iteration: {}".format(epch, i))
        
        set_path = "/set_{}".format(current_set)
        if not os.path.exists(SAVE_LOC + set_path):
            os.makedirs(SAVE_LOC + set_path)

        if i == 0 and epch == 0: # first iteration, create baseline model
            base, base_decomp = lc.extract_weights(model, SAVE_LOC + 
                                                       "/set_{}".format(current_set), DECOMPOSED_LAYERS)
        else:
            if i % 10 == 0: 
                # full snapshot!
                new_model = lazy_restore(base, base_decomp, bias, AlexNet(), 
                                          original.state_dict(), DECOMPOSED_LAYERS, rank = RANK, scaling = SCALING)
                
                original = new_model # Changing previous "original model" used to restore the loRA model.
                
                current_set += 1
                current_iter = 0

                set_path = "/set_{}".format(current_set)
                if not os.path.exists(SAVE_LOC + set_path):
                    os.makedirs(SAVE_LOC + set_path)
                
                # Rebuilding LoRA layers => reset model!
                w, b = getBase(original)
                model = AlexNet_LowRank(w, b, rank = RANK)
                optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
                load_sd_decomp(original.state_dict(), model, DECOMPOSED_LAYERS)
                base, base_decomp = lc.extract_weights(model, SAVE_LOC + 
                                                       "/set_{}".format(current_set), DECOMPOSED_LAYERS)

            else:
                # Delta-compression 
                delta, decomp_delta, bias = lc.generate_delta(base, 
                                                                base_decomp, model.state_dict(), DECOMPOSED_LAYERS)
                compressed_delta, full_delta, compressed_dcomp_delta, full_dcomp_delta  = lc.compress_delta(delta, 
                                                                                                            decomp_delta)
                
                # Saving checkpoint
                lc.save_checkpoint(compressed_delta, compressed_dcomp_delta, bias, current_iter, SAVE_LOC + 
                                "/set_{}".format(current_set))
    
                base = np.add(base, full_delta) # Replace base with latest for delta to accumulate.
                base_decomp = np.add(full_dcomp_delta, base_decomp)

                current_iter += 1
            
        # ==========================
        # Saving using LC-Checkpoint
        # ==========================
        
        if i == 0 and epch == 0:
            cstate = model_original.state_dict()
            set_path = "/set_{}".format(current_set_old_lc)
            if not os.path.exists(SAVE_LOC_OLC + set_path):
                os.makedirs(SAVE_LOC_OLC + set_path)
            # torch.save(cstate, SAVE_LOC_OLC + set_path + "/initial_model.pt")
            # prev_state = olc.extract_weights(model_original)
            prev_state = olc.extract_weights(cstate, SAVE_LOC_OLC + set_path, decomposed_layers = DECOMPOSED_LAYERS)
        else:
            if i % 10 == 0:
                cstate = model_original.state_dict()
                current_set_old_lc += 1
                current_iter_old_lc = 0
                set_path = "/set_{}".format(current_set_old_lc)
                if not os.path.exists(SAVE_LOC_OLC + set_path):
                    os.makedirs(SAVE_LOC_OLC + set_path)
                # torch.save(cstate, SAVE_LOC_OLC + set_path + "/initial_model.pt")
                # prev_state = olc.extract_weights(model_original)
                prev_state = olc.extract_weights(cstate, SAVE_LOC_OLC + set_path, decomposed_layers = DECOMPOSED_LAYERS)
            else:
                cstate = model_original.state_dict()
                old_lc_delta, old_lc_bias = olc.generate_delta(prev_state, cstate, DECOMPOSED_LAYERS)
                olc_compressed_delta, update_prev = olc.compress_data(old_lc_delta)
                olc.save_checkpoint(SAVE_LOC_OLC + "/set_{}".format(current_set_old_lc), olc_compressed_delta, 
                                    old_lc_bias, current_iter_old_lc)
                prev_state = np.add(prev_state, update_prev)
                current_iter_old_lc += 1

        # if i == 0 and epch == 0:
        #     cstate = model_original.state_dict()
        #     set_path = "/set_{}".format(current_set_old_lc)
        #     if not os.path.exists(SAVE_LOC_OLC + set_path):
        #         os.makedirs(SAVE_LOC_OLC + set_path)
        #     torch.save(cstate, SAVE_LOC_OLC + set_path + "/initial_model.pt")
        #     # prev_state = olc.extract_weights(model_original)
        #     prev_state = olc.extract_weights(cstate)
        # else:
        #     if i % 10 == 0:
        #         cstate = model_original.state_dict()
        #         current_set_old_lc += 1
        #         current_iter_old_lc = 0
        #         set_path = "/set_{}".format(current_set_old_lc)
        #         if not os.path.exists(SAVE_LOC_OLC + set_path):
        #             os.makedirs(SAVE_LOC_OLC + set_path)
        #         torch.save(cstate, SAVE_LOC_OLC + set_path + "/initial_model.pt")
        #         # prev_state = olc.extract_weights(model_original)
        #         prev_state = olc.extract_weights(cstate)
        #     else:
        #         cstate = model_original.state_dict()
        #         old_lc_delta, old_lc_bias = olc.generate_delta(prev_state, cstate)
        #         olc_compressed_delta, update_prev = olc.compress_data(old_lc_delta, num_bits = 3)
        #         olc.save_checkpoint(SAVE_LOC_OLC + "/set_{}".format(current_set_old_lc), olc_compressed_delta, 
        #                             old_lc_bias, epch, current_iter_old_lc)
        #         prev_state = np.add(prev_state, update_prev)
        #         current_iter_old_lc += 1
        
        # ==========================
        # Training on Low-Rank Model
        # ==========================

        # Get the inputs and labels
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        print("LoRA+LC Training Loss (Decomposed): {}".format(loss.item()))
        optimizer.step()
            
        # ======================
        # Training on Full Model
        # ======================

        # Zero the parameter gradients
        optimizer_full.zero_grad()

        # Forward + backward + optimize
        outputs_full = model_original(inputs)
        loss_full = torch.nn.functional.cross_entropy(outputs_full,labels)
        loss_full.backward()
        print("LC Training Loss (Full): {}".format(loss_full.item()))
        optimizer_full.step()

        if i % 5 == 0:
            print("Training Accuracy | Decomposed: {}, Full : {}".format(acc(outputs, labels), 
                                                                         acc(outputs_full, labels)))

        if i != 0  and i % 5 == 0: # Evaluation on testing set
            full_accuracy.append(evaluate_accuracy(model_original, test_loader))
            decomposed_full_accuracy.append(evaluate_accuracy(model, test_loader))
            restored_model = lazy_restore(base, base_decomp, bias, AlexNet(), 
                                          original.state_dict(), DECOMPOSED_LAYERS, 
                                          rank = RANK, scaling = SCALING)
            restored_accuracy.append(evaluate_accuracy(restored_model, test_loader))
            restored_lc_model = AlexNet()
            restored_lc_model.load_state_dict(olc.restore_state_dict(prev_state, old_lc_bias, 
                                                                  restored_model.state_dict(), DECOMPOSED_LAYERS))
            lc_accuracy.append(evaluate_accuracy(restored_lc_model, test_loader))
            print("Full accuracy: {}, LC accuracy: {}, Decomposed-Full accuracy: {}, Decomposed-Restored accuracy: {}".format(
                full_accuracy[-1], lc_accuracy[-1], decomposed_full_accuracy[-1], restored_accuracy[-1]))


# In[ ]:


import json

with open(HDFP + "/alexnet-data.json") as f:
    data = json.load(f)
full_accuracy = data['full_acc']
lc_accuracy = data["lc_restored_accuracy"]
restored_accuracy = data["decomposed_restored_accuracy"]
decomposed_full_accuracy = data["decomposed_full_accuracy"]


# In[12]:


# plt.figure(figsize = (30, 5))
# plt.title("AlexNet, Accuracy, Branched @ {} Accuracy".format(BRANCH_ACC))
# plt.plot(full_accuracy, label = "Default AlexNet")
# plt.plot(lc_accuracy, label = "LC AlexNet")
# plt.plot(decomposed_full_accuracy, label = "dLoRA AlexNet")
# plt.plot(restored_accuracy, label = "dLoRA + LC AlexNet")
# plt.xticks([x for x in range(0, 120) if x % 6 == 0], [x for x in range(0, 20)])
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()


# In[13]:


# plt.figure(figsize = (30, 5))
# plt.title("AlexNet, Accuracy, Branched @ {} Accuracy".format(BRANCH_ACC))
# # plt.plot(full_accuracy, label = "Default AlexNet")
# plt.plot(lc_accuracy, label = "LC AlexNet")
# # plt.plot(decomposed_full_accuracy, label = "dLoRA AlexNet")
# # plt.plot(restored_accuracy, label = "dLoRA + LC AlexNet")
# plt.xticks([x for x in range(0, 120) if x % 6 == 0], [x for x in range(0, 20)])
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()


# In[14]:


rangex = [x for x in np.arange(0, 1, 200) if x % 10 == 0]
rangey = [x for x in np.arange(0, 1, 20)]
# plt.figure(figsize = (40, 10))
# ax1 = plt.subplot(1, 2, 1)
# ax1.set_title("AlexNet Absolute Accuracy Loss (Default AlexNet vs LC AlexNet / LC + dLoRA AlexNet)".format(RANK))
# plt.plot(np.abs(np.subtract(np.array(full_accuracy), 
#                      np.array(restored_accuracy))), label = "LC + dLoRA AlexNet")
# plt.plot(np.abs(np.subtract(np.array(full_accuracy), 
#                      np.array(lc_accuracy))), label = "LC AlexNet")
# plt.legend()
# plt.xticks(rangex, rangey)
# plt.ylabel("Absolute Accuracy Loss")
# plt.xlabel("Epoch")
# plt.axhline(y = 0.05, color = 'r')
# plt.ylim(0, 0.5)
# ax2 = plt.subplot(1, 2, 2)
# ax2.set_title("AlexNet Absolute Restoration Accuracy Loss (LC + dLoRA AlexNet & LC AlexNet)")
# plt.plot(np.abs(np.subtract(np.array(restored_accuracy), 
#                      np.array(decomposed_full_accuracy))), label = "LC + dLoRA AlexNet")
# plt.plot(np.abs(np.subtract(np.array(full_accuracy), 
#                      np.array(lc_accuracy))), label = "LC AlexNet")
# plt.legend()
# plt.axhline(y = 0.05, color = 'r')
# plt.ylim(0, 0.5)
# plt.xticks(rangex, rangey)
# plt.ylabel("Absolute Accuracy Loss")
# plt.xlabel("Epoch")
# plt.show()


# In[15]:


import math
def getsize(sl):
    dir = [x for x in os.listdir(sl)]
    csize, usize = 0, 0
    for set in dir:
        for f in os.listdir(sl + "/" + set):
            fp = sl + "/{}/{}".format(set, f)
            csize += os.path.getsize(fp)
            usize += 46.8 * math.pow(2, 20)
    return csize, usize,

def getsize_og_lc(sl):
    dir = [x for x in os.listdir(sl)]
    csize, usize = 0, 0
    for set in dir:
        for f in os.listdir(sl + "/" + set):
            fp = sl + "/{}/{}".format(set, f)
            csize += 1.2 * math.pow(2, 20)
            usize += 46.8 * math.pow(2, 20)
        csize += 46.8 * math.pow(2, 20)
    return csize, usize,


# In[16]:


compressed_size, uncompressed_size = getsize(SAVE_LOC)
a, b = evaluate_compression(uncompressed_size, compressed_size)
compressed_size, uncompressed_size = getsize(SAVE_LOC_OLC)
a1, b1 = evaluate_compression(uncompressed_size, compressed_size)
compressed_size, uncompressed_size = getsize_og_lc(SAVE_LOC_OLC)
a2, b2 = evaluate_compression(uncompressed_size, compressed_size)

print("LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a1, b1))
print("LoRA + LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a, b))
print("LC-Checkpoint")
print("Compression Ratio: {}%, Space Savings: {}%".format(a2, b2))


# In[17]:


import json
data = {
    "full_acc" : full_accuracy,
    "decomposed_restored_accuracy" : restored_accuracy,
    "decomposed_full_accuracy" : decomposed_full_accuracy,
    "lc_restored_accuracy" : lc_accuracy
}
with open(HDFP + "/alexnet-data.json", 'w') as f:
    json.dump(data, f)


# 
