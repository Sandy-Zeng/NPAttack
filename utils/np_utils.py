import torch
import random
import numpy as np
import torchvision.transforms.functional as tF

def construct_gen(model, x, y, x_grid):
    y_hat, z, z_context, r, z_sample = model(x, y, x_grid, x, y)
    return r, z, y_hat

def generate_grid(h, w):
    rows = torch.linspace(0, 1, h).cuda()
    cols = torch.linspace(0, 1, w).cuda()
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid

def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y

def idx_to_x(idx, batch_size, x_grid):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x

def get_context_idx(N, data_dim):
    # generate the indeces of the N context points in a flattened image
    idx = random.sample(range(0, data_dim), N)
    idx = torch.tensor(idx).cuda()
    return idx

class RGB2Y(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return rgb2ycbcr(img)

def rgb2ycbcr(im):
    im = np.array(im)
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return tF.to_pil_image(np.uint8(ycbcr)[:, :, [0]])


def generate_grid_cifar(h, w, c):
    rows = torch.linspace(0, 1, h).cuda()
    cols = torch.linspace(0, 1, w).cuda()
    dim = torch.linspace(0, 1, c).cuda()
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    dim = dim.expand(1, h*w, -1).contiguous().view(1, -1, 1)
    grid = grid.repeat(1, 1, c).view(1, h*w*c, 2)
    grid = torch.cat([grid, dim], dim=-1)
    return grid



