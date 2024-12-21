import sys
import torch
from torch import nn
sys.path.append(r"/home/dcd/zww/repos/auto-encoder-for-cats-master")
__package__ = "src"
from .dataset import CatsDataset
from .unet import UNet
from .utils import contrast

# ! 定义训练的设备
device = torch.device("cuda:2" if torch.cuda.is_available else "cpu")
train_dataloader, val_dataloader, test_dataloader = CatsDataset.get_dataloader(batch_size=1)


mynetwork = UNet(3, 3)
state_dict = torch.load("./tmp/epoch_1.pth")
mynetwork.load_state_dict(state_dict)
mynetwork = mynetwork.to(device)

loss_function = nn.MSELoss()
loss_function = loss_function.to(device)

# 测试
mynetwork.eval()
cnt = 0
with torch.no_grad():
    for data in test_dataloader:
        imgs, _ = data
        imgs = imgs.to(device)
        outputs = mynetwork(imgs)
        loss = loss_function(outputs, imgs)
        print(loss)
        cnt += 1
        # 输出imgs和output图片
        contrast(imgs.cpu(), outputs.cpu())

print(cnt)
