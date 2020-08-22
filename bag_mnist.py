from collections import defaultdict
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []
        #member_dir = glob.glob(self.root+"/*")
        # csvデータの読み出し
        #print(root)
        with open('/Users/hikaru/Desktop/fashionMNIST_PNG/' + root + '_label.csv', 'r') as f:
            f.readline()
            for line in f:
                row = line.strip().split(',')
                # 0:idx, 1:label
                #print('/Users/hikaru/Desktop/fashionMNIST_PNG/' + root + '/bag_' + row[0] + '.png')
                #print('label: ' + row[1])
                self.data.append(cv2.imread('/Users/hikaru/Desktop/fashionMNIST_PNG/' + root + '/bag_' + row[0] + '.png', cv2.IMREAD_GRAYSCALE))
                self.targets.append(int(row[1]))
                #image_dataframe.append([row[1], row[0]])
        #self.image_dataframe = image_dataframe
        # 入力データへの加工
        #self.transform = transform

    # データのサイズ
    def __len__(self):
        return len(self.data)

    # データとラベルの取得
    def __getitem__(self, index):
        #dataframeから画像へのパスとラベルを読み出す
        image, target = self.data[index], self.targets[index]
        #label = self.image_dataframe[idx][0]
        #image_file = '/Users/hikaru/Desktop/fashionMNIST_PNG/train/bag_' + self.image_dataframe[idx][1] + '.png'
        #image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # 稀に別サイズの画像サイズが含まれているので補正
        image = cv2.resize(image, (28, 28))
        
        if self.transform:
            image = self.transform(image)
        return image, target

class TestDataset(Dataset):
    def __init__(self, transform=None):
        # csvデータの読み出し
        image_dataframe = []
        with open('/Users/hikaru/Desktop/fashionMNIST_PNG/test_label.csv', 'r') as f:
            f.readline()
            for line in f:
                row = line.strip().split(',')
                # 0:idx, 1:label
                image_dataframe.append([row[1], row[0]])
        self.image_dataframe = image_dataframe
        # 入力データへの加工
        self.transform = transform
    # データのサイズ
    def __len__(self):
        return len(self.image_dataframe)
    # データとラベルの取得
    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        label = self.image_dataframe[idx][0]
        image_file = '/Users/hikaru/Desktop/fashionMNIST_PNG/test/bag_' + self.image_dataframe[idx][1] + '.png'
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # 稀に別サイズの画像サイズが含まれているので補正
        image = cv2.resize(image, (28, 28))
        
        if self.transform:
            image = self.transform(image)
        return image
        
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1)
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2)
            ,nn.Conv2d(32, 64, kernel_size=3, padding=1)
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout()
            ,nn.Linear(64 * 7 * 7, 128)
            ,nn.ReLU()
            ,nn.Dropout()
            ,nn.Linear(128, 6)
            ,nn.Softmax(dim=1)
            #,nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    
# 学習
def train(model, device, dataloader, optim):
    model.train()

    total_loss = 0
    total_correct = 0
    """
    for batch_idx,(data,target) in enumerate(train_loader):
        optimizer.zero_grad()#Adam初期化
        output=model(data)#model出力
        loss=criterion(output,target)#交差エントロピー誤差
        loss.backward()#逆誤差伝搬
        optimizer.step()#Adam利用
        running_loss+=loss.item()
        if batch_idx%5==4:
            print('[%d,%5d] loss:%.3f'%(epoch,batch_idx+1,running_loss/5))
            train_loss=running_loss/5
            running_loss=0.0
    return train_loss
    """
    for data,target in dataloader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = nn.NLLLoss()(output, target)
        total_loss += float(loss)

        optim.zero_grad()
        loss.backward()

        optim.step()

        pred_target = output.argmax(dim=1)

        total_correct += int((pred_target == target).sum())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    return avg_loss, accuracy

# テスト
def test(model, device, dataloader):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_correct = 0

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = loss = nn.NLLLoss()(output, target)
            total_loss += float(loss)

            pred_target = output.argmax(dim=1)

            total_correct += int((pred_target == target).sum())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    return avg_loss, accuracy
        
def main():
    
    torch.manual_seed(1)
    
    # 作成したデータセットを呼び出し
    train_dataset = MyDataset(root='train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]))
    print(train_dataset.__len__())

    test_dataset = MyDataset(root='test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]))
    print(test_dataset.__len__())

    # 訓練とテストデータ
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # モデルを作成し学習開始
    nll_loss = nn.NLLLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    
    n_epochs = 50

    history = defaultdict(list)
    for epoch in range(n_epochs):
        train_loss, train_accuracy = train(model, device, train_dataloader, optim)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        test_loss, test_accuracy = test(model, device, test_dataloader)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"epoch {epoch+1}"
            f"[train] loss: {train_loss:.6f}, accuracy: {train_accuracy:.0%}"
            f"[test] loss: {test_loss:.6f}, accuracy: {test_accuracy:.0%}"
        )
    # モデルの保存
    torch.save(model.state_dict(), '/Users/hikaru/Desktop/fashionMNIST_PNG/bagMNIST.pt')
    
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(model, dummy_input, '/Users/hikaru/Desktop/fashionMNIST_PNG/bagMNIST.onnx')

    # 学習結果のグラフ表示
    epochs = np.arange(1, n_epochs+1)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    ax1.set_title("Loss")
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.set_title("Accuracy")
    ax2.plot(epochs, history["train_accuracy"], label="train")
    ax2.plot(epochs, history["test_accuracy"], label="test")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.show()
        
if __name__ == '__main__':
    main()