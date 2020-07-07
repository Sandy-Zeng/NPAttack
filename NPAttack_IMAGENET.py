import mnist_mlp
import argparse
from utils.np_utils import *
import random
import os
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import tensorflow as tf
import pretrainedmodels
import time
from ANP.NP_IMAGENET import NP

parser = argparse.ArgumentParser(description='Neural Processes (NP) for Adversarial Attack')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='dataset for training')
parser.add_argument('--data_path', type=str, default= '/data/dataset/ILSVRC2012',
                    help='path to data')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--NP-path', type=str,
                    default='./np_pretrain/imagenet_128/multihead_model_50.pkl',
                    help='the path to load np resnet')
parser.add_argument('--A-path', type=str, default='no',
                    help='the path to load cs matrix')
parser.add_argument('--N', type=int, default=100,
                    help='number of latent vector to be sampled')
parser.add_argument('--I', type=int, default=600,
                    help='max iteration number')
parser.add_argument('--E', type=float, default=0.05,
                    help='max distoration')
parser.add_argument('--LR', type=float, default=0.05,
                    help='max distoration')
parser.add_argument('--images-num', type=int, default=100,
                    help='Test Image')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='targeted attack')
parser.add_argument('--root_path', type=str, default='./log_imagenet',
                    help='path of log file')
parser.add_argument('--type', type=str, default='R',
                    help='type of NPAttack(R, Z and RZ)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def generate_grid(h, w, c):
    rows = torch.linspace(0, 1, h)
    cols = torch.linspace(0, 1, w)
    dim = torch.linspace(0, 1, c)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    dim = dim.expand(1, h * w, -1).contiguous().view(1, -1, 1)
    grid = grid.repeat(1, 1, c).view(1, h * w * c, 2)
    grid = torch.cat([grid, dim], dim=-1)
    return grid


def sample_image_cifar(rec_image, original_image, context_idx, path, file_name):
    # print (context_idx.shape)
    num_examples = min(16, rec_image.shape[0])
    recons = rec_image[:num_examples]
    context_pixels = original_image[:num_examples]
    background = context_pixels
    comparison = torch.cat([background,
                            recons]) + 0.5
    if os.path.exists(path) == False:
        os.mkdir(path)
    save_image(comparison.cpu(), '%s/reconstruct_%s_%s' % (path, args.dataset, file_name) + '.png', nrow=num_examples)
    save_image(comparison.cpu(), '%s/reconstruct_%s_%s' % (path, args.dataset, file_name) + '.pdf', nrow=num_examples)


def upsample(y_rec, y_ori_rec, ori_input, eps):
    noise = torch.clamp(y_rec - y_ori_rec, -eps, eps)
    up_noise = F.interpolate(noise, size=(224, 224, 3), mode='bilinear')
    adv_input = ori_input + up_noise
    return adv_input


def save(input, idx, name, pred, label):
    global root_path
    # print(input.shape)
    input = input.permute(0, 2, 3, 1).squeeze(0)
    # input = input.permute(2, 0, 1)
    input = input.numpy()
    input = (input + 0.5) * 255.0
    # print(input.shape)
    img = Image.fromarray(input.astype(np.uint8))
    if target_attack:
        path = '%s/Adv_ImageNet_%d_%.2f_targeted' % (root_path, args.N, args.E)
    else:
        path = '%s/Adv_ImageNet_%d_%.2f_untarget' % (root_path, args.N, args.E)
    if os.path.exists(path) == False:
        os.makedirs(path)
    img.save(path + '/%s_%d_%d_%d.png' % (name, idx, pred, label))


def get_context(y_all, x_grid, model):
    R = []
    MU = []
    S = []
    # print(y_all.shape)
    for xs in range(int(resize_dim / 32)):
        for ys in range(int(resize_dim / 32)):
            # print(y_all.shape)
            data_dim = 32 * 32 * 3
            y_all_new = y_all[:, :, xs * 32:xs * 32 + 32, ys * 32:ys * 32 + 32]

            y_all_new = y_all_new.to(device).permute(0, 2, 3, 1).to('cuda:1').contiguous().view(batch_size, -1, 1)

            x_context = x_grid.expand(batch_size, -1, -1).to('cuda:1')
            y_context = y_all_new

            y_mean_new, sigma, z_all_mu, z_all_var, z_con_mu, z_con_var, s_context = model(x_context, y_context,
                                                                                           x_context, y_context)

            mu1 = z_con_mu.unsqueeze(1).expand(-1, data_dim, -1)
            S1 = z_con_var.unsqueeze(1).expand(-1, data_dim, -1)

            R.append(s_context.cpu().detach())
            MU.append(mu1.cpu().detach())
            S.append(S1.cpu().detach())
    MU = torch.cat(MU, dim=1)
    S = torch.cat(S, dim=1)
    R = torch.cat(R, dim=1)
    # print(MU.shape)
    return R, MU, S


def recons(model, MU, S, R, x_grid):
    start = time.time()
    idx = 0
    y_rec = []
    channel_dim = 299
    c = 3
    data_dim = 32 * 32 * 3
    num = MU.shape[0]
    x_context = x_grid.expand(batch_size, -1, -1).to('cuda:1')
    y_rec = torch.zeros((num, 3, resize_dim, resize_dim))
    for xs in range(int(resize_dim / 32)):
        for ys in range(int(resize_dim / 32)):
            mu1 = MU[:, idx * data_dim: idx * data_dim + data_dim, :]
            S1 = S[:, idx * data_dim: idx * data_dim + data_dim, :]
            s_context = R[:, idx * data_dim:idx * data_dim + data_dim, :].to('cuda:1')
            idx = idx + 1

            z_op = torch.distributions.normal.Normal(mu1, S1).rsample().to('cuda:1')
            y_ori_rec = model.decoder(s_context, z_op, x_context.expand(num, -1, -1))[0]
            y_ori_rec = y_ori_rec.cpu().detach()

            y_rec[:, :, xs * 32:xs * 32 + 32, ys * 32:ys * 32 + 32] = y_ori_rec[:num].view(-1, 32, 32, 3).permute(0, 3,
                                                                                                                  1, 2)
    y_rec = F.interpolate(y_rec, size=(299, 299), mode='bilinear', align_corners=True)
    # print(y_rec.shape)
    end = time.time()
    # print('Rec Time:', end - start)
    return y_rec.cpu().detach()


def decode(y_all, model):
    batch_size = 1
    data_dim = 299 * 299 * 3
    y_all = y_all.permute(0, 2, 3, 1).to('cuda:1').contiguous().view(batch_size, -1, 1)
    x_grid = generate_grid(299, 299, 3).to('cuda:1')
    x_context = x_grid.expand(batch_size, -1, -1).to('cuda:1')
    y_context = y_all

    y_mean_new, sigma, z_all_mu, z_all_var, z_con_mu, z_con_var, s_context = model(x_context, y_context, x_context,
                                                                                   y_context)
    mu1 = z_con_mu.unsqueeze(1).expand(-1, data_dim, -1)
    S1 = z_con_var.unsqueeze(1).expand(-1, data_dim, -1)

    z_op = torch.distributions.normal.Normal(mu1, S1).rsample().to('cuda:1')
    y_ori_rec = model.decoder(s_context, z_op, x_context)[0]
    save(y_ori_rec.cpu().detach(), 1, 'test', 0, 0)


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    # print((1 + x) / (1 - x))
    # assert False
    return (np.log((1 + x) / (1 - x))) * 0.5


def NP_Attack(model, resize_input, ori_input, y_label, target, N, attack_model, target_attack=False):
    """NP(ANP) estimator"""
    # use Neural Process in prediction mode
    # set batch size to 1
    eps = args.E
    nb_iter = args.I
    lr = 0.05
    context_idx = get_context_idx(data_dim, data_dim).to('cuda:1')

    R, MU, S = get_context(resize_input, x_grid, model)

    y_ori_rec = recons(model, MU, S, R, x_grid)
    # save(y_ori_rec, attack_success, 'ori_rec', y_label.item(), y_label.item())
    # save(resize_input.cpu().detach(), attack_success, 'resize', y_label.item(), y_label.item())
    # save(ori_input.cpu().detach(), attack_success, 'ori', y_label.item(), y_label.item())
    # assert False

    newimg = torch_arctanh((ori_input.cpu()) / 0.5)
    for iter in range(nb_iter):
        y_rec = recons(model, MU, S, R, x_grid)
        # save(y_rec.cpu().detach(), attack_success, 'rec', iter, y_label.item())
        noise = y_rec - y_ori_rec
        # noise = torch.clamp(y_rec - y_ori_rec, -eps, eps)
        # save(noise.cpu().detach(), attack_success, 'noise', iter, y_label.item())

        adv_input = torch.tanh(newimg.cpu() + noise.cpu()) * 0.5
        realdist = torch.clamp(adv_input - torch.tanh(newimg) * 0.5, -eps, eps)
        adv_input = realdist + torch.tanh(newimg) * 0.5

        adv_input = adv_input.to('cuda:0')
        pred_logit = F.softmax(attack_model(adv_input)).cpu().detach().numpy()
        y_pred = np.reshape(np.argmax(pred_logit, axis=1), y_label.shape)
        if target_attack:
            print('Prediction: ', y_pred.item(), 'Label:', y_label.item(), 'Target', target, 'Confidence',
                  pred_logit[0, target])
        else:
            print('Prediction: ', y_pred.item(), 'Label:', y_label.item(), 'Confidence',
                  pred_logit[0, y_label.item()])
        print(np.sort(pred_logit[0])[-1:-4:-1])

        if target_attack:
            if y_pred.item() == target:
                print('Attack Success: Predict: %d Real: %d' % (y_pred.item(), y_label.item()))
                save(adv_input.cpu().detach(), attack_success, 'adv', y_pred.item(), y_label.item())
                save(ori_input.cpu().detach(), attack_success, 'ori', y_label.item(), y_label.item())
                save((adv_input-ori_input).cpu().detach(), attack_success, 'noise', y_pred.item(), y_label.item())
                perturbation = realdist.cpu()
                return perturbation, True, iter, y_pred.item()
        else:
            if y_pred.item() != y_label.item():
                print('Attack Success: Predict: %d Real: %d' % (y_pred.item(), y_label.item()))
                # if cnt < log_time:
                save(adv_input.cpu().detach(), attack_success, 'adv', y_pred.item(), y_label.item())
                save(ori_input.cpu().detach(), attack_success, 'ori', y_label.item(), y_label.item())
                save((adv_input-ori_input).cpu().detach(), attack_success, 'noise', y_pred.item(), y_label.item())
                perturbation = realdist.cpu()
                return perturbation, True, iter, y_pred.item()

        b = time.time()

        if args.type == 'R':
            pertur_R = torch.randn((N, R.shape[1], R.shape[2])).float()
            e = time.time()
            print('Sample Time:', e - b)
            R_pertur = R.expand(N, -1, -1) + pertur_R * 0.1
            y_rec = recons(model, MU.expand(N, -1, -1), S.expand(N, -1, -1), R_pertur, x_grid)
        elif args.type == 'Z':
            pertur = torch.randn((N, MU.shape[1], MU.shape[2])).float()
            e = time.time()
            print('Sample Time:', e - b)
            mu1_perturb = MU + pertur * 0.1
            y_rec = recons(model, mu1_perturb, S.expand(N, -1, -1), R.expand(N, -1, -1), x_grid)
        elif args.type == 'RZ':
            # perturb r z
            pertur = torch.randn((N, MU.shape[1], MU.shape[2])).float()
            pertur_R = torch.randn((N, R.shape[1], R.shape[2])).float()
            e = time.time()
            print('Sample Time:', e - b)
            mu1_perturb = MU + pertur * 0.1
            R_pertur = R.expand(N, -1, -1) + pertur_R * 0.1
            y_rec = recons(model, mu1_perturb, S.expand(N, -1, -1), R_pertur, x_grid)

        noise = y_rec - y_ori_rec.expand(N, -1, -1, -1)
        adv_input = torch.tanh(newimg.expand(N, -1, -1, -1) + noise.cpu()) * 0.5
        realdist = torch.clamp(adv_input - torch.tanh(newimg.expand(N, -1, -1, -1)) * 0.5, -eps, eps)
        adv_input = realdist + torch.tanh(newimg.expand(N, -1, -1, -1)) * 0.5

        adv_input = adv_input.to('cuda:0')
        pred_logit = F.softmax(attack_model(adv_input), dim=1).cpu().detach()
        if target_attack:
            loss, confidence = CW_loss(pred_logit, target, True)
        else:
            loss, confidence = CW_loss(pred_logit, y_label, False)
        adv_loss = - loss.view(N, 1)
        A = (adv_loss - torch.mean(adv_loss)) / (torch.std(adv_loss) + 1e-7)
        pgd_loss = - torch.mean(adv_loss)

        if iter % 1 == 0:
            print('Iter %d Adv LOSS %.4f ' % (iter, pgd_loss))
            # print(np.mean(np.sort(pred_logit), 0)[-1:-4:-1])

        if iter % 100 == 0 and iter != 0 and lr > 5e-5:
            lr = lr / 2

        # update the latent variable
        if args.type == 'R':
            R_loss = torch.matmul(torch.transpose(pertur_R.view(N, -1), 0, 1), A).view(MU.shape)
            R = R + lr / (N * 0.1) * R_loss.cpu().detach()
        if args.type == 'Z':
            NES_loss = torch.matmul(torch.transpose(pertur.view(N, -1), 0, 1), A).view(MU.shape)
            MU = MU + lr / (N * 0.1) * NES_loss.cpu().detach()
        if args.type == 'RZ':
            R_loss = torch.matmul(torch.transpose(pertur_R.view(N, -1), 0, 1), A).view(MU.shape)
            R = R + lr / (N * 0.1) * R_loss.cpu().detach()
            NES_loss = torch.matmul(torch.transpose(pertur.view(N, -1), 0, 1), A).view(MU.shape)
            MU = MU + lr / (N * 0.1) * NES_loss.cpu().detach()


    y_hat = recons(model, MU, S, R, x_grid)
    perturbation = torch.clamp(y_hat - y_ori_rec, -eps, eps).cpu()
    return perturbation, False, 0, y_label.item()


def CW_loss(pred, y_label, targeted=False):
    pred = torch.log(pred)
    target_onehot = torch.zeros((1, 1000))
    target_onehot[0][y_label] = 1
    y_pred = torch.sum(target_onehot * pred, dim=1)
    other = torch.max((1. - target_onehot) * pred - target_onehot * 10000., dim=1)[0]
    # print (other.shape)
    confidence = torch.mean(y_pred)
    if targeted == False:
        loss = y_pred - other
    else:
        loss = other - y_pred
    # loss = torch.clamp(y_pred - other, 0., 1000)
    return loss, confidence


def load_attack_model():
    model_name = 'inceptionv3'  # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = model.to('cuda:0')
    model.eval()
    return model


def load_NP_model(NP_path, A_path):
    # NP_model = NP(args).to(device)
    NP_model = torch.load(NP_path).to('cuda:1')
    # NP_model = NP_model.cuda()
    NP_model = NP_model.module
    A_Matrix = None
    print('Load NP Model')
    return NP_model, A_Matrix


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    for i in range(samples):
        if targeted:
            if inception:
                # for inception, randomly choose 10 target classes
                seq = np.random.choice(range(1, 1001), 1)
                # seq = [580] # grand piano
            else:
                # for CIFAR and MNIST, generate all target classes
                seq = range(data.test_labels.shape[1])

            # print ('image label:', np.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(data.test_labels[start + i])
                true_ids.append(start + i)
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])
            labels.append(data.test_labels[start + i])
            true_ids.append(start + i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)

    return inputs, targets, labels, true_ids


if args.dataset == 'imagenet':
    # Data loading code
    data_path = args.data_path
    # traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    channel_dim = 32
    resize_dim = 128
    c = 3
    data_dim = channel_dim * channel_dim * 3
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])
    transformers = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transformers),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)
    data_dim = channel_dim * channel_dim * 3

attack_success = 0
Test_Image = 0.
attack_success_images = []
test_images = []
avg_iters = 0.
log_time = 1
cnt = 0
avg_L1 = 0.
avg_L2 = 0.
avg_Lp = 0.

target_attack = args.targeted

NP_model, A_Matrix = load_NP_model(args.NP_path, None)
attack_model = load_attack_model()
x_grid = generate_grid(channel_dim, channel_dim, c).to('cuda:1')

context_idx = get_context_idx(data_dim, data_dim).to('cuda:1')

# start = 0
# end = all_inputs.shape[0]
root_path = args.root_path

if os.path.exists(root_path) == False:
    os.mkdir(root_path)

resize_transform = transforms.Compose([
        transforms.Resize((resize_dim,resize_dim)),
        transforms.ToTensor(),
        normalize,
    ])


real_test_image = 0.
for batch_idx, (y_all, y_label) in enumerate(val_loader):
    # print(y_all.shape)
    y_all = y_all.to('cuda:0')
    pred = F.softmax(attack_model(y_all), dim=1).cpu().detach().numpy()
    y_pred = np.argmax(pred, axis=1)
    # print(y_pred.item(), y_label.item())
    if y_pred.item() != y_label.item():
        continue
    # attack_model = None
    Test_Image += 1
    real_test_image += 1
    batch_size = y_all.shape[0]

    # resize
    y_all_resize = y_all.permute(0, 2, 3, 1)
    y_all_resize = y_all_resize.squeeze(0).cpu().numpy()
    resize_input = Image.fromarray(((y_all_resize+0.5)*255.0).astype(np.uint8))
    resize_input = resize_transform(resize_input)

    # resize_input = np.resize(y_all_resize, (3, 64, 64))
    resize_input = torch.Tensor(resize_input).unsqueeze(0)

    # random target
    index_list = np.concatenate([np.arange(y_label.item()), np.arange(y_label.item() + 1, 1000)])
    random_target = np.random.choice(index_list)
    perturbation, attack, iters, pred = NP_Attack(NP_model, resize_input, y_all, y_label, random_target, args.N, attack_model,
                                                  target_attack)

    if attack == True:
        attack_success += 1
        avg_iters += iters
        perturbation = perturbation.detach().numpy()
        avg_L1 += np.sum(np.abs(perturbation))
        avg_L2 += np.sum((perturbation) ** 2) ** .5
        avg_Lp += np.max(np.abs(perturbation))
        print('Predict %d, Label %d' % (pred, y_label.item()))
        # ASR = float(attack_success) / Test_Image
        ASR = float(attack_success) / real_test_image
        AI = avg_iters / float(attack_success)
        log_msg = 'Attacking %d Images Attack Success Rate: %.4f Avg Iter: %.4f Avg Query: %d Avg L1 %.4f Avg L2 %.4f Avg Lp %.4f \n ' % \
                  (Test_Image, ASR, AI, AI * args.N, avg_L1 / attack_success, avg_L2 / attack_success,
                   avg_Lp / attack_success)
        print(log_msg)
        if attack_success == 1:
            with open('%s/log_%s_%.3f_%d_%.4f_%d_%d.txt' % (
            root_path, args.dataset, args.E, args.N, args.LR, args.I, target_attack), 'w') as f:
                f.write(log_msg)
        else:
            with open('%s/log_%s_%.3f_%d_%.4f_%d_%d.txt' % (
            root_path, args.dataset, args.E, args.N, args.LR, args.I, target_attack), 'a') as f:
                f.write(log_msg)
    if Test_Image > args.images_num:
        break

ASR = float(attack_success) / Test_Image
AI = avg_iters / float(attack_success)
print('Attack Success Rate: %.4f' % (ASR))
print('Average Iteration: %.4f:' % (AI))
log_msg = 'Attacking %d Images Attack Success Rate: %.4f Avg Iter: %.4f Avg Query: %d Avg L1 %.4f Avg L2 %.4f Avg Lp %.4f \n ' % \
          (Test_Image, ASR, AI, AI * args.N, avg_L1 / attack_success, avg_L2 / attack_success,
           avg_Lp / attack_success)
with open('%s/log_%s_%.3f_%d_%.4f_%d_%d.txt' % (
root_path, args.dataset, args.E, args.N, args.LR, args.I, target_attack), 'a') as f:
    f.write(log_msg)
f.close()












