import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN
from datasets import CaptchaData


# 配置gpu训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 配置transform
transforms = Compose([
    ToTensor()
])


# 加载卷积神经网络
cnn = CNN()
cnn.to(device)


#超参数
batch_size = 64
lr = 0.001
Epoch = 50


# 设置优化器和损失函数
optimizer = optim.Adam(cnn.parameters(), lr=lr)
loss_func = nn.MultiLabelSoftMarginLoss()


# acc 准确率计算
def calculat_acc(output, target):
    output, target = output.view(-1, 62), target.view(-1, 62)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


# 拟合，利用train判断是 训练集还是验证集
def fit(model, loader, train = True):
    if train:
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()

    acc_history = []
    loss_history = []
    for img, target in tqdm(loader, leave=False):
        if train:
            # 梯度清零
            optimizer.zero_grad()
        img = img.to(device)
        target = target.to(device)
        output = cnn(img)
        loss = loss_func(output, target)

        if train:
            # 优化
            loss.backward()
            optimizer.step()
        acc = calculat_acc(output, target)
        acc_history.append(float(acc))
        loss_history.append(float(loss))
    if train:
        print('train_loss: {:.4}|train_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))
    else:
        print('validation_loss: {:.4}|validation_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))

    return torch.mean(torch.tensor(loss_history)), torch.mean(torch.tensor(acc_history))

def train():
    # 读取训练集
    train_dataset = CaptchaData('dataset/train/', transform=transforms)

    # 按照1:5的比例随机抽取验证集
    train_size = int(len(train_dataset) * 0.8)
    validation_size = int(len(train_dataset) - train_size)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

    # dataloader 加载数据集
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(Epoch):

        train_loss, train_acc = fit(cnn, train_data_loader, train=True)
        val_loss, val_acc = fit(cnn, validation_data_loader, train=False)
        train_loss_list.append(train_loss.cpu())
        val_loss_list.append(val_loss.cpu())
        train_acc_list.append(train_acc.cpu())
        val_acc_list.append(val_acc.cpu())

        print('Epoch', epoch + 1, '| train_loss: %.4f' % train_loss, '|train_acc:%.4f' % train_acc, '| validation_loss: %.4f' % val_loss, '|validation_acc:%.4f' % val_acc)

    torch.save(cnn.state_dict(), "./model.pth")

    plt.plot(train_loss_list, "b", label="train_loss")
    plt.plot(val_loss_list, "g", label="test_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    plt.plot(train_acc_list, "b", label="train_acc")
    plt.plot(val_acc_list, "g", label="test_acc")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.savefig("train_result_{}_Epoch.png".format(Epoch), dpi=600)
    plt.show()

if __name__ == "__main__":
    train()
