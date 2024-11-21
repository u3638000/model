import sys
sys.path.append(".")


import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def set_seed(seed=1):  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def plot_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list):

    plt.figure(figsize=(10, 6))

    # train loss
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
    # val loss
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # train acc
    plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Train Acc')
    # val acc
    plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.show()

####### Define the Dataset.
class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):

        self.label_name = {"c0": 0, "c1": 1, "c2": 2, "c3": 3}
        self.data_info = self.get_img_info(data_dir)  # data_info stores all image paths and tags, and reads samples through index in DataLoader
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # categories
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # images
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info

def build_dataloader(INPUT_SIZE, BATCH_SIZE):
    # dataset path
    split_dir = os.path.join("C:\Workspace\DASC7606\cat_dog_car_bike")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "val")
    
    # dataset mean and std
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomCrop(INPUT_SIZE, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    
    # Build MyDataset instance
    train_data = MyDataset(data_dir=train_dir, transform=train_transform)
    valid_data = MyDataset(data_dir=valid_dir, transform=valid_transform)
    
    # Build DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    
    return train_loader, valid_loader

def train_model(model, optimizer, criterion, train_loader, epoch, max_epoch):

    loss_train = 0.
    correct = 0.
    total = 0.

    model.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        loss_train += loss.item()

        # update parameters
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()

    
    loss_train_epoch = loss_train / len(train_loader)

    print("Train: Epoch[{:0>3}/{:0>3}] Iterations[{:0>3}/{:0>3}] Loss: {:.4f} Accuracy: {:.2%}".format(
        epoch, max_epoch, i+1, len(train_loader), loss_train_epoch, correct / total))

    train_loss = loss_train_epoch
    train_acc = correct / total

    return  train_loss, train_acc

def valid_model(model, criterion, valid_loader, epoch, max_epoch):

    correct_val = 0.
    total_val = 0.
    loss_val = 0.
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

            loss_val += loss.item()

    # print val loss
    loss_val_epoch = loss_val / len(valid_loader)
    print("Val: Epoch[{:0>3}/{:0>3}] Iterations[{:0>3}/{:0>3}] Loss: {:.4f} Accuracy: {:.2%}".format(
        epoch, max_epoch, j+1, len(valid_loader), loss_val_epoch, correct_val / total_val))

    val_loss = loss_val_epoch
    val_acc = correct_val / total_val
    return  val_loss, val_acc

def main(model, LR, INPUT_SIZE=32, BATCH_SIZE=20, MAX_EPOCH=30):

    val_interval = 1
    
    ################# Build the Dataloader.
    train_loader,  valid_loader = build_dataloader(INPUT_SIZE, BATCH_SIZE)
    
    ################# Bulid Model.
    criterion = nn.CrossEntropyLoss()                                       # loss
    optimizer = optim.RMSprop(model.parameters(), lr=LR)                    # optimizer
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # scheduler
    
    ################# Plot the loss and acc.
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    ################# Main training process.
    for epoch in range(MAX_EPOCH):
        train_loss, train_acc = train_model(model, optimizer, criterion, train_loader, epoch, MAX_EPOCH)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
    
        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = valid_model(model, criterion, valid_loader, epoch, MAX_EPOCH)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
    
    # plot_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(            
            nn.Conv2d(3,32,(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 32*112*112
        self.conv1 = nn.Sequential(
            nn.Conv2d(32,64,(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 64*56*56
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 128*28*28
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 256*14*14
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 512*7*7
        self.fc = nn.Sequential(
            nn.Linear(512*7*7,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,4)
        )
        self._initialize_weights()

    def forward(self,x):
        x = x.to(device)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
LR = 0.001
model = Net().to(device)
main(model, LR, INPUT_SIZE=224, BATCH_SIZE=20, MAX_EPOCH=90)

torch.save(model, 'modelQ5.h1')

"""
Train: Epoch[000/090] Iterations[084/084] Loss: 52.7481 Accuracy: 61.43%
Val: Epoch[000/090] Iterations[042/042] Loss: 6.1945 Accuracy: 72.57%
Train: Epoch[001/090] Iterations[084/084] Loss: 5.6884 Accuracy: 70.27%
Val: Epoch[001/090] Iterations[042/042] Loss: 2.1544 Accuracy: 61.56%
Train: Epoch[002/090] Iterations[084/084] Loss: 2.5791 Accuracy: 72.18%
Val: Epoch[002/090] Iterations[042/042] Loss: 0.7070 Accuracy: 76.89%
Train: Epoch[003/090] Iterations[084/084] Loss: 1.4185 Accuracy: 73.31%
Val: Epoch[003/090] Iterations[042/042] Loss: 0.6951 Accuracy: 73.89%
Train: Epoch[004/090] Iterations[084/084] Loss: 0.9916 Accuracy: 72.18%
Val: Epoch[004/090] Iterations[042/042] Loss: 0.6063 Accuracy: 77.25%
Train: Epoch[005/090] Iterations[084/084] Loss: 0.7292 Accuracy: 74.87%
Val: Epoch[005/090] Iterations[042/042] Loss: 0.6931 Accuracy: 77.49%
Train: Epoch[006/090] Iterations[084/084] Loss: 0.5441 Accuracy: 75.94%
Val: Epoch[006/090] Iterations[042/042] Loss: 0.5071 Accuracy: 80.48%
Train: Epoch[007/090] Iterations[084/084] Loss: 0.5496 Accuracy: 76.24%
Val: Epoch[007/090] Iterations[042/042] Loss: 0.4535 Accuracy: 79.40%
Train: Epoch[008/090] Iterations[084/084] Loss: 0.6650 Accuracy: 76.66%
Val: Epoch[008/090] Iterations[042/042] Loss: 1.7328 Accuracy: 70.78%
Train: Epoch[009/090] Iterations[084/084] Loss: 0.8574 Accuracy: 75.28%
Val: Epoch[009/090] Iterations[042/042] Loss: 0.4606 Accuracy: 79.64%
Train: Epoch[010/090] Iterations[084/084] Loss: 0.5297 Accuracy: 78.15%
Val: Epoch[010/090] Iterations[042/042] Loss: 0.7690 Accuracy: 70.78%
Train: Epoch[011/090] Iterations[084/084] Loss: 0.4373 Accuracy: 80.12%
Val: Epoch[011/090] Iterations[042/042] Loss: 0.4391 Accuracy: 81.44%
Train: Epoch[012/090] Iterations[084/084] Loss: 0.5519 Accuracy: 80.54%
Val: Epoch[012/090] Iterations[042/042] Loss: 0.9102 Accuracy: 80.24%
Train: Epoch[013/090] Iterations[084/084] Loss: 0.4993 Accuracy: 80.54%
Val: Epoch[013/090] Iterations[042/042] Loss: 0.5363 Accuracy: 81.44%
Train: Epoch[014/090] Iterations[084/084] Loss: 0.4378 Accuracy: 81.61%
Val: Epoch[014/090] Iterations[042/042] Loss: 0.3753 Accuracy: 82.87%
Train: Epoch[015/090] Iterations[084/084] Loss: 0.4193 Accuracy: 82.21%
Val: Epoch[015/090] Iterations[042/042] Loss: 0.4272 Accuracy: 82.51%
Train: Epoch[016/090] Iterations[084/084] Loss: 0.4530 Accuracy: 83.34%
Val: Epoch[016/090] Iterations[042/042] Loss: 0.6628 Accuracy: 76.29%
Train: Epoch[017/090] Iterations[084/084] Loss: 0.3708 Accuracy: 83.22%
Val: Epoch[017/090] Iterations[042/042] Loss: 0.3508 Accuracy: 83.47%
Train: Epoch[018/090] Iterations[084/084] Loss: 0.3827 Accuracy: 84.18%
Val: Epoch[018/090] Iterations[042/042] Loss: 0.5273 Accuracy: 82.99%
Train: Epoch[019/090] Iterations[084/084] Loss: 0.2988 Accuracy: 87.28%
Val: Epoch[019/090] Iterations[042/042] Loss: 1.0135 Accuracy: 80.00%
Train: Epoch[020/090] Iterations[084/084] Loss: 0.3606 Accuracy: 86.51%
Val: Epoch[020/090] Iterations[042/042] Loss: 0.3871 Accuracy: 86.11%
Train: Epoch[021/090] Iterations[084/084] Loss: 0.3912 Accuracy: 85.79%
Val: Epoch[021/090] Iterations[042/042] Loss: 0.4111 Accuracy: 82.51%
Train: Epoch[022/090] Iterations[084/084] Loss: 0.4140 Accuracy: 86.69%
Val: Epoch[022/090] Iterations[042/042] Loss: 0.5291 Accuracy: 82.51%
Train: Epoch[023/090] Iterations[084/084] Loss: 0.2872 Accuracy: 89.43%
Val: Epoch[023/090] Iterations[042/042] Loss: 0.3427 Accuracy: 85.75%
Train: Epoch[024/090] Iterations[084/084] Loss: 0.2279 Accuracy: 91.46%
Val: Epoch[024/090] Iterations[042/042] Loss: 0.3720 Accuracy: 86.35%
Train: Epoch[025/090] Iterations[084/084] Loss: 0.2813 Accuracy: 89.85%
Val: Epoch[025/090] Iterations[042/042] Loss: 0.3962 Accuracy: 86.35%
Train: Epoch[026/090] Iterations[084/084] Loss: 0.2607 Accuracy: 90.03%
Val: Epoch[026/090] Iterations[042/042] Loss: 0.3958 Accuracy: 85.51%
Train: Epoch[027/090] Iterations[084/084] Loss: 0.2705 Accuracy: 90.45%
Val: Epoch[027/090] Iterations[042/042] Loss: 0.3550 Accuracy: 86.35%
Train: Epoch[028/090] Iterations[084/084] Loss: 0.2416 Accuracy: 91.04%
Val: Epoch[028/090] Iterations[042/042] Loss: 0.6452 Accuracy: 85.27%
Train: Epoch[029/090] Iterations[084/084] Loss: 0.2385 Accuracy: 92.06%
Val: Epoch[029/090] Iterations[042/042] Loss: 0.4601 Accuracy: 83.35%
Train: Epoch[030/090] Iterations[084/084] Loss: 0.2004 Accuracy: 92.90%
Val: Epoch[030/090] Iterations[042/042] Loss: 0.3143 Accuracy: 88.02%
Train: Epoch[031/090] Iterations[084/084] Loss: 0.2274 Accuracy: 91.52%
Val: Epoch[031/090] Iterations[042/042] Loss: 0.3241 Accuracy: 88.86%
Train: Epoch[032/090] Iterations[084/084] Loss: 0.2571 Accuracy: 92.12%
Val: Epoch[032/090] Iterations[042/042] Loss: 0.4086 Accuracy: 88.74%
Train: Epoch[033/090] Iterations[084/084] Loss: 0.1760 Accuracy: 93.55%
Val: Epoch[033/090] Iterations[042/042] Loss: 0.4565 Accuracy: 86.35%
Train: Epoch[034/090] Iterations[084/084] Loss: 0.1717 Accuracy: 94.15%
Val: Epoch[034/090] Iterations[042/042] Loss: 0.3376 Accuracy: 88.50%
Train: Epoch[035/090] Iterations[084/084] Loss: 0.1914 Accuracy: 93.43%
Val: Epoch[035/090] Iterations[042/042] Loss: 0.3440 Accuracy: 88.86%
Train: Epoch[036/090] Iterations[084/084] Loss: 0.1416 Accuracy: 94.33%
Val: Epoch[036/090] Iterations[042/042] Loss: 0.3806 Accuracy: 88.86%
Train: Epoch[037/090] Iterations[084/084] Loss: 0.1511 Accuracy: 94.39%
Val: Epoch[037/090] Iterations[042/042] Loss: 0.3710 Accuracy: 88.26%
Train: Epoch[038/090] Iterations[084/084] Loss: 0.1751 Accuracy: 94.81%
Val: Epoch[038/090] Iterations[042/042] Loss: 0.5883 Accuracy: 85.99%
Train: Epoch[039/090] Iterations[084/084] Loss: 0.1648 Accuracy: 94.45%
Val: Epoch[039/090] Iterations[042/042] Loss: 0.4142 Accuracy: 88.86%
Train: Epoch[040/090] Iterations[084/084] Loss: 0.1309 Accuracy: 95.40%
Val: Epoch[040/090] Iterations[042/042] Loss: 0.3754 Accuracy: 89.58%
Train: Epoch[041/090] Iterations[084/084] Loss: 0.1522 Accuracy: 95.04%
Val: Epoch[041/090] Iterations[042/042] Loss: 0.3666 Accuracy: 89.82%
Train: Epoch[042/090] Iterations[084/084] Loss: 0.1822 Accuracy: 94.21%
Val: Epoch[042/090] Iterations[042/042] Loss: 0.4058 Accuracy: 88.50%
Train: Epoch[043/090] Iterations[084/084] Loss: 0.1067 Accuracy: 95.52%
Val: Epoch[043/090] Iterations[042/042] Loss: 0.3154 Accuracy: 90.06%
Train: Epoch[044/090] Iterations[084/084] Loss: 0.1152 Accuracy: 95.88%
Val: Epoch[044/090] Iterations[042/042] Loss: 0.3834 Accuracy: 90.30%
Train: Epoch[045/090] Iterations[084/084] Loss: 0.1157 Accuracy: 96.66%
Val: Epoch[045/090] Iterations[042/042] Loss: 0.5268 Accuracy: 88.26%
Train: Epoch[046/090] Iterations[084/084] Loss: 0.1065 Accuracy: 95.94%
Val: Epoch[046/090] Iterations[042/042] Loss: 0.3485 Accuracy: 89.94%
Train: Epoch[047/090] Iterations[084/084] Loss: 0.0908 Accuracy: 96.84%
Val: Epoch[047/090] Iterations[042/042] Loss: 0.4286 Accuracy: 89.82%
Train: Epoch[048/090] Iterations[084/084] Loss: 0.1684 Accuracy: 95.76%
Val: Epoch[048/090] Iterations[042/042] Loss: 1.0997 Accuracy: 85.15%
Train: Epoch[049/090] Iterations[084/084] Loss: 0.0954 Accuracy: 95.58%
Val: Epoch[049/090] Iterations[042/042] Loss: 0.3494 Accuracy: 90.66%
Train: Epoch[050/090] Iterations[084/084] Loss: 0.0594 Accuracy: 97.85%
Val: Epoch[050/090] Iterations[042/042] Loss: 0.3881 Accuracy: 89.10%
Train: Epoch[051/090] Iterations[084/084] Loss: 0.0647 Accuracy: 97.55%
Val: Epoch[051/090] Iterations[042/042] Loss: 0.4351 Accuracy: 90.66%
Train: Epoch[052/090] Iterations[084/084] Loss: 0.0841 Accuracy: 96.84%
Val: Epoch[052/090] Iterations[042/042] Loss: 0.3934 Accuracy: 90.90%
Train: Epoch[053/090] Iterations[084/084] Loss: 0.0989 Accuracy: 97.49%
Val: Epoch[053/090] Iterations[042/042] Loss: 0.5103 Accuracy: 89.46%
Train: Epoch[054/090] Iterations[084/084] Loss: 0.1018 Accuracy: 96.48%
Val: Epoch[054/090] Iterations[042/042] Loss: 0.4964 Accuracy: 90.30%
Train: Epoch[055/090] Iterations[084/084] Loss: 0.1175 Accuracy: 96.48%
Val: Epoch[055/090] Iterations[042/042] Loss: 0.5682 Accuracy: 87.66%
Train: Epoch[056/090] Iterations[084/084] Loss: 0.0650 Accuracy: 97.91%
Val: Epoch[056/090] Iterations[042/042] Loss: 0.5127 Accuracy: 91.02%
Train: Epoch[057/090] Iterations[084/084] Loss: 0.0547 Accuracy: 97.85%
Val: Epoch[057/090] Iterations[042/042] Loss: 0.5156 Accuracy: 91.02%
Train: Epoch[058/090] Iterations[084/084] Loss: 0.1105 Accuracy: 97.31%
Val: Epoch[058/090] Iterations[042/042] Loss: 0.4491 Accuracy: 90.42%
Train: Epoch[059/090] Iterations[084/084] Loss: 0.0764 Accuracy: 97.79%
Val: Epoch[059/090] Iterations[042/042] Loss: 0.6128 Accuracy: 89.34%
Train: Epoch[060/090] Iterations[084/084] Loss: 0.0706 Accuracy: 97.67%
Val: Epoch[060/090] Iterations[042/042] Loss: 0.4351 Accuracy: 91.02%
Train: Epoch[061/090] Iterations[084/084] Loss: 0.0916 Accuracy: 96.90%
Val: Epoch[061/090] Iterations[042/042] Loss: 1.2065 Accuracy: 86.83%
Train: Epoch[062/090] Iterations[084/084] Loss: 0.0625 Accuracy: 97.97%
Val: Epoch[062/090] Iterations[042/042] Loss: 0.6556 Accuracy: 89.94%
Train: Epoch[063/090] Iterations[084/084] Loss: 0.0848 Accuracy: 97.79%
Val: Epoch[063/090] Iterations[042/042] Loss: 0.5203 Accuracy: 90.78%
Train: Epoch[064/090] Iterations[084/084] Loss: 0.0513 Accuracy: 98.33%
Val: Epoch[064/090] Iterations[042/042] Loss: 0.5518 Accuracy: 90.78%
Train: Epoch[065/090] Iterations[084/084] Loss: 0.0739 Accuracy: 98.09%
Val: Epoch[065/090] Iterations[042/042] Loss: 0.6020 Accuracy: 89.70%
Train: Epoch[066/090] Iterations[084/084] Loss: 0.0841 Accuracy: 97.67%
Val: Epoch[066/090] Iterations[042/042] Loss: 0.4462 Accuracy: 92.22%
Train: Epoch[067/090] Iterations[084/084] Loss: 0.0224 Accuracy: 99.34%
Val: Epoch[067/090] Iterations[042/042] Loss: 0.5948 Accuracy: 91.14%
Train: Epoch[068/090] Iterations[084/084] Loss: 0.0508 Accuracy: 97.85%
Val: Epoch[068/090] Iterations[042/042] Loss: 0.4923 Accuracy: 91.26%
Train: Epoch[069/090] Iterations[084/084] Loss: 0.0435 Accuracy: 98.69%
Val: Epoch[069/090] Iterations[042/042] Loss: 0.5660 Accuracy: 92.22%
Train: Epoch[070/090] Iterations[084/084] Loss: 0.0596 Accuracy: 98.69%
Val: Epoch[070/090] Iterations[042/042] Loss: 0.5574 Accuracy: 90.66%
Train: Epoch[071/090] Iterations[084/084] Loss: 0.0399 Accuracy: 98.39%
Val: Epoch[071/090] Iterations[042/042] Loss: 0.6018 Accuracy: 91.74%
Train: Epoch[072/090] Iterations[084/084] Loss: 0.0494 Accuracy: 98.39%
Val: Epoch[072/090] Iterations[042/042] Loss: 0.6075 Accuracy: 90.42%
Train: Epoch[073/090] Iterations[084/084] Loss: 0.0501 Accuracy: 98.45%
Val: Epoch[073/090] Iterations[042/042] Loss: 0.6603 Accuracy: 90.66%
Train: Epoch[074/090] Iterations[084/084] Loss: 0.0230 Accuracy: 98.99%
Val: Epoch[074/090] Iterations[042/042] Loss: 0.6732 Accuracy: 90.06%
Train: Epoch[075/090] Iterations[084/084] Loss: 0.0759 Accuracy: 98.45%
Val: Epoch[075/090] Iterations[042/042] Loss: 1.3369 Accuracy: 86.47%
Train: Epoch[076/090] Iterations[084/084] Loss: 0.0697 Accuracy: 98.33%
Val: Epoch[076/090] Iterations[042/042] Loss: 0.6216 Accuracy: 92.10%
Train: Epoch[077/090] Iterations[084/084] Loss: 0.0332 Accuracy: 98.99%
Val: Epoch[077/090] Iterations[042/042] Loss: 0.6421 Accuracy: 91.02%
Train: Epoch[078/090] Iterations[084/084] Loss: 0.0248 Accuracy: 99.16%
Val: Epoch[078/090] Iterations[042/042] Loss: 1.7301 Accuracy: 83.83%
Train: Epoch[079/090] Iterations[084/084] Loss: 0.1033 Accuracy: 98.09%
Val: Epoch[079/090] Iterations[042/042] Loss: 0.5023 Accuracy: 90.42%
Train: Epoch[080/090] Iterations[084/084] Loss: 0.0326 Accuracy: 98.87%
Val: Epoch[080/090] Iterations[042/042] Loss: 0.7032 Accuracy: 90.90%
Train: Epoch[081/090] Iterations[084/084] Loss: 0.0421 Accuracy: 98.93%
Val: Epoch[081/090] Iterations[042/042] Loss: 0.5297 Accuracy: 91.62%
Train: Epoch[082/090] Iterations[084/084] Loss: 0.0236 Accuracy: 99.34%
Val: Epoch[082/090] Iterations[042/042] Loss: 0.5517 Accuracy: 91.98%
Train: Epoch[083/090] Iterations[084/084] Loss: 0.0323 Accuracy: 98.87%
Val: Epoch[083/090] Iterations[042/042] Loss: 0.6285 Accuracy: 91.62%
Train: Epoch[084/090] Iterations[084/084] Loss: 0.0176 Accuracy: 99.28%
Val: Epoch[084/090] Iterations[042/042] Loss: 0.5609 Accuracy: 90.54%
Train: Epoch[085/090] Iterations[084/084] Loss: 0.0365 Accuracy: 99.28%
Val: Epoch[085/090] Iterations[042/042] Loss: 0.4878 Accuracy: 91.50%
Train: Epoch[086/090] Iterations[084/084] Loss: 0.0548 Accuracy: 98.39%
Val: Epoch[086/090] Iterations[042/042] Loss: 1.6497 Accuracy: 87.07%
Train: Epoch[087/090] Iterations[084/084] Loss: 0.0539 Accuracy: 98.69%
Val: Epoch[087/090] Iterations[042/042] Loss: 0.6552 Accuracy: 92.10%
Train: Epoch[088/090] Iterations[084/084] Loss: 0.0387 Accuracy: 99.28%
Val: Epoch[088/090] Iterations[042/042] Loss: 1.1526 Accuracy: 88.50%
Train: Epoch[089/090] Iterations[084/084] Loss: 0.0201 Accuracy: 99.52%
Val: Epoch[089/090] Iterations[042/042] Loss: 0.7424 Accuracy: 91.02%
"""
