import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as tF
from torch.nn import init
from torch.autograd import Variable
import time


def rgb2ycbcr(im):
    im = np.array(im)
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return tF.to_pil_image(np.uint8(ycbcr)[:, :, [0]])


class RGB2Y(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        return rgb2ycbcr(img)


parser = argparse.ArgumentParser(description='Neural Processes (NP) for MNIST image completion')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset for training')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--r_dim', type=int, default=128, metavar='N',
                    help='dimension of r, the hidden representation of the context points')
parser.add_argument('--z_dim', type=int, default=128, metavar='N',
                    help='dimension of z, the global latent variable')
parser.add_argument('--hidden_dim', type=int, default=400, metavar='N',
                    help='dimension of z, the global latent variable')
parser.add_argument('--att_type', type=str, default='multihead',
                    help='attention type')
parser.add_argument('--rep_type', type=str, default='identity',
                    help='representation type')
parser.add_argument('--restore_model', type=str, default='np_pretrain/cifar_128',
                    help='restore resnet path')
parser.add_argument('--measurement_type', type=str, default='gaussian',
                    help='the type of measurement matrix A')
parser.add_argument('--log_dir', type=str, default='./log/cifar_np_multihead_128',
                    help='path to log')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

if args.dataset == 'mnist':
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],
                                                                                 std=[1])])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    channel_dim, data_dim = 28, 784
    c = 1
elif args.dataset == 'cifar10':
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                 std=[1, 1, 1])])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/zengyuyuan/data', train=True, download=True,
                         transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/zengyuyuan/data', train=False, transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    channel_dim = 32
    c = 3
    data_dim = channel_dim * channel_dim * 3


def get_context_idx():
    # generate the indeces of the N context points in a flattened image
    idx = np.array(range(0, data_dim))
    idx = torch.tensor(idx, device=device)
    return idx


# def generate_grid(h, w, c):
#     rows = torch.linspace(0, 1, h, device=device)
#     cols = torch.linspace(0, 1, w, device=device)
#     dim = torch.linspace(0, 1, c, device=device)
#     dim = dim.view(c, 1).repeat(1, h * w).view(1, -1, 1)
#     grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
#     grid = grid.unsqueeze(0)
#     grid = grid.repeat(1, 3, 1)
#     grid = torch.cat([grid, dim], dim=-1)
#     # grid = torch.stack([grid.repeat(1, c, 1), dim], dim=2)
#     # dim = dim.expand(1, h*w, -1).contiguous().view(1, -1, 1)
#     # grid = torch.cat([grid.repeat(1, c, 1), dim], dim=-1)
#     return grid

def generate_grid(h, w, c):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    dim = torch.linspace(0, 1, c, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    dim = dim.expand(1, h * w, -1).contiguous().view(1, -1, 1)
    grid = grid.repeat(1, 1, c).view(1, h * w * c, 2)
    grid = torch.cat([grid, dim], dim=-1)
    return grid


def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, batch_size):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x


class NP(nn.Module):
    def __init__(self, args):
        super(NP, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.hidden_dim = args.hidden_dim

        self.h_1 = nn.Linear(c + 1, self.hidden_dim)
        self.h_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_3 = nn.Linear(self.hidden_dim, self.r_dim)

        self.s_dense = nn.Linear(self.r_dim, self.r_dim)

        self.s_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.s_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)

        self.h_4 = nn.Linear(c + 1, self.hidden_dim)
        self.h_5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_6 = nn.Linear(self.hidden_dim, self.r_dim)
        self.s = nn.Linear(self.r_dim, self.z_dim)

        self.h_7 = nn.Linear(c, self.hidden_dim)
        self.h_8 = nn.Linear(self.hidden_dim, c)

        self.h_9 = nn.Linear(c, self.hidden_dim)
        self.h_10 = nn.Linear(self.hidden_dim, c)

        if args.rep_type == 'mlp':
            self.att_rep_k = nn.Linear(self.z_dim, self.z_dim)
            self.att_rep_k = nn.Linear(self.z_dim, self.z_dim)

        self.num_heads = 4

        if args.att_type == 'multihead':
            d_k, d_v = c, self.r_dim
            self.heads = []
            head_size = int(self.r_dim / self.num_heads)

            for i in range(self.num_heads):
                self.heads.append(nn.Conv1d(d_k, d_k, 1))
                self.heads[-1].weight.data.normal_(0, d_k ** (-0.5))
                self.heads.append(nn.Conv1d(d_k, d_k, 1))
                self.heads[-1].weight.data.normal_(0, d_k ** (-0.5))
                self.heads.append(nn.Conv1d(d_v, head_size, 1))
                self.heads[-1].weight.data.normal_(0, d_k ** (-0.5))
                self.heads.append(nn.Conv1d(head_size, self.r_dim, 1))
                self.heads[-1].weight.data.normal_(0, d_v ** (-0.5))

            self.heads = nn.Sequential(*(self.heads))

        self.g_1 = nn.Linear(self.z_dim * 2 + c, self.hidden_dim)
        self.g_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.g_5 = nn.Linear(self.hidden_dim, c)
        self.g_mean = nn.Linear(self.hidden_dim, 1)
        self.g_var = nn.Linear(self.hidden_dim, 1)

        self.attention_type = args.att_type

    def encoder_determinate(self, x_y):
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(x_y))
        x_y = F.relu(self.h_3(x_y))
        return x_y

    def crossAtt_x(self, x):
        # print('x-----------------',x.shape)
        x = F.relu(self.h_7(x))
        # print('x-----------------',x.shape)
        x = F.relu(self.h_8(x))
        # print('x-----------------',x.shape)
        return x

    def crossAtt_xTarget(self, x_target):
        x_target = F.relu(self.h_9(x_target))
        x_target = F.relu(self.h_10(x_target))
        return x_target

    def encoder_latent(self, x_y):
        # print('h--------------',x_y.shape)
        x_y = F.relu(self.h_4(x_y))
        # print('h_4---------------',x_y.shape)
        x_y = F.relu(self.h_5(x_y))
        # print('h_5---------------',x_y.shape)
        x_y = F.relu(self.h_6(x_y))
        # print('h_6---------------',x_y.shape)
        return x_y

    def dot_product_attention(self, q, k, v, normalise):
        """Computes dot product attention.

        Args:
          q: queries. tensor of  shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          normalise: Boolean that determines whether weights sum to 1.

        Returns:
          tensor of shape [B,m,d_v].
        """
        # print (q.shape, k.shape, v.shape)
        d_k = q.shape[-1]
        scale = np.sqrt(1.0 * d_k)
        unnorm_weights = torch.einsum('bjk,bik->bij', (k, q)) / scale  # [B,m,n]
        if normalise:
            weight_fn = nn.functional.softmax
        else:
            weight_fn = torch.sigmoid

        weights = weight_fn(unnorm_weights, dim=-1)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', (weights, v))  # [B,m,d_v]
        return rep

    def laplace_attention(self, q, k, v, scale, normalise):
        """Computes laplace exponential attention.

        Args:
          q: queries. tensor of shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          scale: float that scales the L1 distance.
          normalise: Boolean that determines whether weights sum to 1.

        Returns:
          tensor of shape [B,m,d_v].
        """
        k = k.unsqueeze(dim=1)  # [B,1,n,d_k]
        q = q.unsqueeze(dim=2)  # [B,m,1,d_k]
        unnorm_weights = -((k - q) / scale).abs()  # [B,m,n,d_k]
        unnorm_weights = unnorm_weights.sum(dim=-1)  # [B,m,n]
        if normalise:
            weight_fn = nn.functional.softmax
        else:
            weight_fn = lambda x: 1 + nn.functional.tanh(x)
        weights = weight_fn(unnorm_weights, dim=-1)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', (weights, v))  # [B,m,d_v]
        return rep

    def multihead_attention(self, q, k, v, num_heads=8):
        """Computes multi-head attention.

        Args:
          q: queries. tensor of  shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          num_heads: number of heads. Should divide d_v.

        Returns:
          tensor of shape [B,m,d_v].
        """
        # print (q.shape)
        # print (k.shape)
        # print (v.shape)
        d_k = q.shape[-1]
        d_v = v.shape[-1]
        head_size = d_v / num_heads

        rep = 0.0
        for h in range(num_heads):
            o = self.dot_product_attention(
                self.heads[h * 4](q.permute(0, 2, 1)).permute(0, 2, 1),
                self.heads[h * 4 + 1](k.permute(0, 2, 1)).permute(0, 2, 1),
                self.heads[h * 4 + 2](v.permute(0, 2, 1)).permute(0, 2, 1),
                normalise=True)

            rep += self.heads[h * 4 + 3](o.permute(0, 2, 1)).permute(0, 2, 1)
        return rep

    def corss_attention(self, content_x, target_x, r):
        if self.attention_type == 'uniform':
            return torch.mean(r, dim=1).unsqueeze(1).expand(-1, data_dim, -1)
        elif self.attention_type == 'laplace':
            return self.laplace_attention(target_x, content_x, r, 1, True)
        elif self.attention_type == 'dot':
            return self.dot_product_attention(target_x, content_x, r, True)
        elif self.attention_type == 'multihead':
            return self.multihead_attention(target_x, content_x, r, self.num_heads)

    def self_attention(self, input_xy):
        """
            inputs :
                input_xy : input feature maps [B,C,W,H]
            returns :
                out : self attention value + input feature
                attention: [B,N,N,N](N is Width*Height)
        """
        #  print(input_xy.shape)
        input_xy = input_xy.permute(0, 2, 1)
        # print('input_xy=-----------------',input_xy.shape)
        m_batchsize, C, width_height = input_xy.size()
        q_conv = nn.Conv1d(in_channels=C, out_channels=C // 8, kernel_size=1)
        k_conv = nn.Conv1d(in_channels=C, out_channels=C // 8, kernel_size=1)
        v_conv = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1)
        gamma = nn.Parameter(torch.zeros(1)).to(device)

        q_conv = q_conv.to(device)
        k_conv = k_conv.to(device)
        v_conv = v_conv.to(device)
        q = q_conv(input_xy)
        # print('q-------------------',q.shape)
        q = q.view(m_batchsize, -1, width_height).permute(0, 2, 1)
        k = k_conv(input_xy).view(m_batchsize, -1, width_height)
        energy = torch.bmm(q, k)
        attention = nn.functional.softmax(energy, dim=-1)
        v = v_conv(input_xy).view(m_batchsize, -1, width_height)

        rep = torch.bmm(v, attention.permute(0, 2, 1))
        rep = rep.view(m_batchsize, width_height, C)
        rep = rep.permute(0, 2, 1)
        # print('rep--------------',rep.device)
        # print('input_xy---------------------',input_xy.device)
        # print('gamma--------------------',gamma.device)
        rep = gamma * rep + input_xy

        rep = rep.permute(0, 2, 1)
        return rep, attention

    def reparameterise(self, z):
        mu, var = z

        m = torch.distributions.normal.Normal(mu, var)
        z_sample = m.rsample()

        z_sample = z_sample.unsqueeze(1).expand(-1, data_dim, -1)

        return z_sample

    def decoder(self, r_sample, z_sample, x_target):  # decoder
        # print (r_sample.shape)
        # print (z_sample.shape)
        # print (x_target.shape)
        z_x = torch.cat([r_sample, z_sample, x_target], dim=2)

        input = F.relu(self.g_1(z_x))
        input = F.relu(self.g_2(input))
        input = F.relu(self.g_3(input))
        input = F.relu(self.g_4(input))

        y_mean = self.g_mean(input)
        y_var = self.g_var(input)
        sigma = 0.1 + 0.9 * F.softplus(y_var)

        y_dis = torch.distributions.normal.Normal(y_mean, sigma)

        y_hat = y_mean
        return (y_hat, y_dis)

    def xy_to_z(self, x, y):  # latent path of encoder
        # print('x--------------',x.shape)
        # print('y--------------',y.shape)
        x_y = torch.cat([x, y], dim=2)
        # print(x_y.shape)
        input_xy = self.encoder_latent(x_y)
        # print(input_xy.shape)
        s_i, _ = self.self_attention(input_xy)
        s = torch.mean(s_i, dim=1)

        s = F.relu(self.s_dense(s))
        mu = self.s_to_z_mean(s)
        logvar = self.s_to_z_logvar(s)
        var = (0.1 + 0.9 * torch.sigmoid(logvar)) * 0.1

        return mu, var

    def xy_to_r(self, x, y, x_target):  # deterministic path of encoder
        x_y = torch.cat([x, y], dim=2)
        input_xy = self.encoder_determinate(x_y)
        r_i, _ = self.self_attention(input_xy)

        x = self.crossAtt_x(x)
        x_target = self.crossAtt_xTarget(x_target)
        r = self.corss_attention(x, x_target, r_i)

        return self.s(r)

    def forward(self, x_context, y_context, x_all=None, y_all=None):
        x_target = x_grid.expand(y_context.shape[0], -1, -1)

        z_context = self.xy_to_z(x_context, y_context)  # (mu, logvar) of z
        # print (z_context.shape)
        r_context = self.xy_to_r(x_context, y_context, x_target)

        if self.training:  # loss function will try to keep z_context close to z_all
            z_all = self.xy_to_z(x_all, y_all)
            r_all = r_context
        else:  # at test time we don't have the image so we use only the context
            z_all = z_context
            r_all = r_context

        z_sample = self.reparameterise(z_all)
        r_sample = r_all

        # reconstruct the whole image including the provided context points
        y_hat = self.decoder(r_sample, z_sample, x_target)

        return y_hat, z_all, z_context, r_sample, z_sample


def kl_div_gaussians(mu_q, var_q, mu_p, var_p):
    # var_p = torch.exp(logvar_p)
    logvar_p, logvar_q = torch.log(var_p), torch.log(var_q)
    kl_div = (var_q + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum() / kl_div.shape[0]
    return kl_div


def np_loss(y_hat, y, z_all, z_context):
    y_hat, y_dis = y_hat
    log_p = y_dis.log_prob(y).sum(dim=-1).sum(dim=-1)
    BCE = - log_p.sum() / log_p.shape[0]
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD


################################################################################
model = NP(args).to(device)
# model = torch.load('/home/baiyang/zengyuyuan/NP/np_pretrain/cifar_512/multihead_model.pkl')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
x_grid = generate_grid(channel_dim, channel_dim, c)
os.makedirs("results/", exist_ok=True)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (y_all, _) in enumerate(train_loader):
        batch_size = y_all.shape[0]
        if args.dataset == 'mnist':
            y_all = y_all.to(device).view(batch_size, -1, 1)
        elif args.dataset == 'cifar10':
            y_all = y_all.permute(0, 2, 3, 1).to(device).view(batch_size, -1, 1)
        elif args.dataset == 'celeba':
            y_all = y_all.permute().to(device).view(batch_size, -1, c)
        elif args.dataset == 'streetview':
            y_all = y_all.permute().to(device).view(batch_size, -1, c)

        # N = random.randint(1, data_dim)  # number of context points
        context_idx = get_context_idx()

        x_context = x_grid.expand(batch_size, -1, -1)
        # y_context = idx_to_y(context_idx, y_all)
        y_context = y_all

        x_all = x_grid.expand(batch_size, -1, -1)

        optimizer.zero_grad()
        y_hat, z_all, z_context, _, _ = model(x_context, y_context, x_all, y_all)
        # print(y_hat[0].shape)

        loss = np_loss(y_hat, y_all, z_all, z_context)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(y_all), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(y_all)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    if os.path.exists(args.log_dir) == False:
        os.makedirs(args.log_dir)
    filename = os.path.join(args.log_dir, 'train.txt')
    with open(filename, 'a') as f:
        f.write('====> Epoch: {} Average loss: {:.4f}\n'.format(
            epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (y_all, _) in enumerate(test_loader):

            batch_size = y_all.shape[0]
            if args.dataset == 'mnist':
                y_all = y_all.to(device).view(batch_size, -1, 1)
            elif args.dataset == 'cifar10':
                y_all = y_all.to(device).permute(0, 2, 3, 1).to(device).contiguous().view(batch_size, -1, 1)

            # context_idx = get_context_idx()
            x_context = x_grid.expand(batch_size, -1, -1)
            # x_context = idx_to_x(context_idx, batch_size)
            # y_context = idx_to_y(context_idx, y_all)
            y_context = y_all

            y_hat, z_all, z_context, _, _ = model(x_context, y_context)
            test_loss += np_loss(y_hat, y_all, z_all, z_context).item()

            if i == 0:  # save PNG of reconstructed examples
                num_examples = min(batch_size, 16)

                recons = []
                context_idx = get_context_idx()
                # x_context = idx_to_x(context_idx, batch_size)
                # y_context = idx_to_y(context_idx, y_all)
                y_hat, _, _, _, _ = model(x_context, y_context)
                y_hat = y_hat[0]
                recons = y_hat[:num_examples].view(-1, channel_dim, channel_dim, c).permute(0, 3, 1, 2)

                background = torch.tensor([0., 0., 1.], device=device)
                background = background.view(1, -1, 1).expand(num_examples, 3, data_dim).contiguous()

                if args.dataset == 'mnist':
                    context_pixels = y_all[:num_examples].view(num_examples, 1, -1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                            recons]) + 0.5
                elif args.dataset == 'cifar10':
                    context_pixels = y_all[:num_examples].view(-1, channel_dim, channel_dim, 3).permute(0, 3, 1, 2)
                    background = context_pixels
                    comparison = torch.cat([background,
                                            recons]) + 0.5
                elif args.dataset == 'celeba':
                    context_pixels = y_all[:num_examples].permute(0, 2, 1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                            recons]) + 0.5
                elif args.dataset == 'streetview':
                    context_pixels = y_all[:num_examples].permute(0, 2, 1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                            recons]) + 0.5
                save_image(comparison.cpu(),
                           args.log_dir + '/%s_ep_' % (args.dataset + '_' + args.att_type) + str(epoch) + '.png',
                           nrow=num_examples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    filename = os.path.join(args.log_dir, 'test.txt')
    with open(filename, 'a') as f:
        f.write('====> Test set loss: {:.4f}\n'.format(test_loss))


if os.path.exists(args.restore_model) == False:
    os.makedirs(args.restore_model)
if os.path.exists(args.log_dir) == False:
    os.makedirs(args.log_dir)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    t = time.time()
    torch.save(model, os.path.join(args.restore_model, args.att_type + '_model.pkl'))







