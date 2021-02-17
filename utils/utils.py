import matplotlib.pyplot as plt
import os
import torch
import numpy as np


def plot(data, data_avg, title, file_name, output_directory, legend=None):
    plt.plot(data)
    plt.plot(data_avg)
    if legend is not None:
        plt.legend(legend, loc='upper left')
    plt.title(title)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig("{}/{}.pdf".format(output_directory, file_name))
    plt.clf()


def preprocess_ppo(x, prev_x):
    return torch.cat([state_to_tensor(x), state_to_tensor(prev_x)], dim=1)


def state_to_tensor(x):
    """ preprocessing 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
    if x is None:
        return torch.zeros(1, 9300)
    x = x[:, 7:193]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    x = x[::2, ::2, 0]  # downsample by factor of 2.
    x[x == 58] = 0  # erase background (background type 1)
    x[x == 43] = 0  # erase background (background type 2)
    x[x == 48] = 0  # erase background (background type 3)
    x[x != 0] = 1  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    pp_x = torch.from_numpy(x.astype(np.float32).ravel()).unsqueeze(0)
    return pp_x
