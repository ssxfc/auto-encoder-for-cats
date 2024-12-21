import os

import torch
import matplotlib.pyplot as plt

import uuid


def contrast(img: torch.FloatTensor, pred_mask: torch.LongTensor):
    r"""画出原图，真实掩码以及各个类别的预测掩码

    Args:
        img: 原图，size为(batch, channel, h, w)
        pred_mask: 模型输出的掩码，size为(batch, channel, h, w)
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_title('input')
    ax[0].imshow(img[0].permute(1, 2, 0).numpy())
    ax[1].set_title(f'outputs')

    ax[1].imshow(pred_mask[0].permute(1, 2, 0).numpy())
    plt.savefig(f'./tmp/contrast/{uuid.uuid1()}.png')
    plt.close()
