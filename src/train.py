import sys
import torch
import time
from torch import nn
sys.path.append(r"/home/dcd/zww/repos/auto-encoder-for-cats-master")
__package__ = "src"
from .dataset import CatsDataset
from .unet import UNet

# ! 定义训练的设备
device = torch.device("cuda:2" if torch.cuda.is_available else "cpu")
train_dataloader, val_dataloader, test_dataloader = CatsDataset.get_dataloader(batch_size=8)


mynetwork = UNet(3, 3)
# ! .to(device)
mynetwork = mynetwork.to(device)

loss_function = nn.MSELoss()
loss_function = loss_function.to(device)


learning_rate = 1e-4
optimizer = torch.optim.SGD(mynetwork.parameters(), lr=learning_rate)

epoch = 100
start_time = time.time()

for i in range(epoch):
    # 训练
    print(f"----------第 {i} 轮训练开始----------")
    mynetwork.train()
    cnt = 0
    for data in train_dataloader:
        imgs, _ = data
        imgs = imgs.to(device)

        outputs = mynetwork(imgs)
        loss = loss_function(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        if cnt % 100 == 0:
            print(loss)
    # 验证
    mynetwork.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, _ = data
            imgs = imgs.to(device)
            outputs = mynetwork(imgs)
            loss = loss_function(outputs, imgs)
            total_test_loss += loss
    print(f"epoch: {i} 验证集上的Loss == {total_test_loss}")
    torch.save(mynetwork.state_dict(), f"tmp/epoch_{i}.pth")
