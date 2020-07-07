import argparse
import os
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import mnist_mlp
from utils.np_utils import *
from wrn.WRN import Network as WRN

parser = argparse.ArgumentParser(description='Neural Processes (NP) for Adversarial Attack')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default='cifar-10',
                    help='dataset for training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-path', type=str, default='/data/zengyuyuan/data',
                    help='path to data')
parser.add_argument('--NP-path', type=str, default='./np_pretrain',
                    help='the path to load np resnet')
parser.add_argument('--A-path', type=str, default='no',
                    help='the path to load cs matrix')
parser.add_argument('--N', type=int, default=30,
                    help='number of latent vector to be sampled')
parser.add_argument('--I', type=int, default=200,
                    help='max iteration number')
parser.add_argument('--E', type=float, default=0.031,
                    help='max distoration')
parser.add_argument('--LR', type=float, default=0.01,
                    help='max distoration')
parser.add_argument('--images-num', type=int, default=200,
                    help='Test Image')
parser.add_argument('--attack-path', type=str, default='./wrn',
                    help='the path of attack resnet')
parser.add_argument('--model', type=str, default='wrn',
                    help='type of attack model')
parser.add_argument('--root_path', type=str, default='./log',
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

def sample_image(rec_image, original_image, context_idx, path, file_name):
    # print (context_idx.shape)
    num_examples = min(16,rec_image.shape[0])
    recons = []
    img = rec_image
    recons.append(img[:num_examples])
    recons = torch.cat(recons).permute(0, 2, 1).view(-1, c, channel_dim, channel_dim).expand(-1, 3, -1, -1)
    background = torch.tensor([0., 0., 1.])
    background = background.view(1, -1, 1).expand(num_examples, 3, data_dim).contiguous()

    context_pixels = original_image[:num_examples].view(num_examples, 1, -1)[:, :, context_idx]
    context_pixels = context_pixels.expand(num_examples, 3, -1)
    background[:, :, context_idx] = context_pixels
    comparison = (torch.cat([background.view(-1, 3, channel_dim, channel_dim), recons]) + 0.5)
    if os.path.exists(path) == False:
        os.mkdir(path)

    save_image(comparison.cpu(), '%s/reconstruct_%s_%s' % (path,args.dataset, file_name) + '.png', nrow=num_examples)
    save_image(comparison.cpu(), '%s/reconstruct_%s_%s' % (path,args.dataset, file_name) + '.pdf', nrow=num_examples)

def sample_image_cifar(rec_image, original_image, context_idx, path, file_name):
    # print (context_idx.shape)
    num_examples = min(16,rec_image.shape[0])
    recons = rec_image[:num_examples]
    context_pixels = original_image[:num_examples]
    background = context_pixels
    comparison = torch.cat([background,
                            recons]) + 0.5
    if os.path.exists(path) == False:
        os.mkdir(path)
    save_image(comparison.cpu(), '%s/reconstruct_%s_%s' % (path, args.dataset, file_name) + '.png', nrow=num_examples)
    save_image(comparison.cpu(), '%s/reconstruct_%s_%s' % (path, args.dataset, file_name) + '.pdf', nrow=num_examples)


def NP_Attack(model, x_input, y_label, N, attack_model):
    """NP(ANP) estimator"""
    # use Neural Process in prediction mode
    #set batch size to 1
    eps = args.E
    nb_iter = args.I
    lr = args.LR
    x_input = x_input.to(device)
    context_idx = get_context_idx(data_dim, data_dim)
    x_context = idx_to_x(context_idx, batch_size, x_grid)
    y_context = idx_to_y(context_idx, x_input)
    #s_context refer to r
    #z_context refer to the Guassian distribution of z
    s_context, z_context, y_pred = construct_gen(model, x_context, y_context, x_grid)
    # mu and sigma of Gaussian Distribution of z
    mu1 = z_context[0].unsqueeze(1).expand(-1, data_dim, -1)
    S1 = z_context[1].unsqueeze(1).expand(-1, data_dim, -1)
    mu1 = torch.tensor(mu1, requires_grad=False)
    S1 = torch.tensor(S1, requires_grad=False)
    #sample z
    z_op = torch.distributions.normal.Normal(mu1, S1).rsample()
    z_op = Variable(z_op, requires_grad=True)
    z_shape = z_op.shape

    x_target = x_grid.expand(y_context.shape[0], -1, -1)
    # print(s_context)

    if args.dataset == 'cifar-10':
        x_input = x_input.view(-1, channel_dim, channel_dim, c).permute(0, 3, 1, 2)
    for iter in range(nb_iter):
        z_op = torch.distributions.normal.Normal(mu1, S1).rsample()
        y_rec = model.decoder(s_context, z_op, x_target)[0]
        if args.dataset == 'cifar-10':
            y_rec = y_rec.view(-1, channel_dim, channel_dim, c).permute(0, 3, 1, 2)
        y_rec = x_input + torch.clamp(y_rec - x_input, -eps, eps)

        pred_logit = F.softmax(attack_model(y_rec), dim=1).cpu().detach()
        y_pred = np.reshape(np.argmax(pred_logit.numpy(), axis=1), y_label.shape)
        print(pred_logit.numpy()[0][y_label.item()])
        print('probs:', np.sort(pred_logit.numpy()[0])[-1:-4:-1])
        if y_pred.item() != y_label.item():
            print ('Attack Success: Predict: %d Real: %d'% (y_pred.item(), y_label.item()))
            if cnt < log_time:
                attack_success_images.append(y_rec.cpu())
                test_images.append(x_input.cpu())
            perturbation = torch.clamp(y_rec - x_input, -eps, eps).cpu()
            return perturbation, True, iter, y_pred.item()

        if args.type == 'R':
            pertur_s = torch.from_numpy(np.random.randn(N, s_context.shape[1], s_context.shape[2])).float()
            s_context_pertur = s_context + pertur_s.to(device) * S1.expand(N, -1, -1)
            z_op = torch.distributions.normal.Normal(mu1, S1).rsample()
            y_pred = model.decoder(s_context_pertur, z_op.expand(N, -1, -1), x_target.expand(N, -1, -1))[0]
        elif args.type == 'Z':
            # print(args.type)
            pertur = torch.from_numpy(np.random.randn(N, mu1.shape[1], mu1.shape[2])).float()
            mu1_perturb = mu1 + pertur.to(device) * S1.expand(N, -1, -1)
            z_perturb = torch.distributions.normal.Normal(mu1_perturb, S1.expand(N, -1, -1)).rsample()
            z_perturb = Variable(z_perturb, requires_grad=True)
            y_pred = model.decoder(s_context.expand(N, -1, -1), z_perturb, x_target.expand(N, -1, -1))[0]
        elif args.type == 'RZ':
            #perturb r
            pertur_s = torch.from_numpy(np.random.randn(N, s_context.shape[1], s_context.shape[2])).float()
            s_context_pertur = s_context + pertur_s.to(device) * S1.expand(N, -1, -1)
            #perturb z
            pertur = torch.from_numpy(np.random.randn(N, mu1.shape[1], mu1.shape[2])).float()
            mu1_perturb = mu1 + pertur.to(device) * S1.expand(N, -1, -1)
            z_perturb = torch.distributions.normal.Normal(mu1_perturb, S1.expand(N, -1, -1)).rsample()
            z_perturb = Variable(z_perturb, requires_grad=True)
            y_pred = model.decoder(s_context_pertur, z_perturb, x_target.expand(N, -1, -1))[0]

        if args.dataset == 'cifar-10':
            y_pred = y_pred.view(-1, channel_dim, channel_dim, c).permute(0, 3, 1, 2)
        y_pred = x_input + torch.clamp(y_pred - x_input, -eps, eps)

        pred_logit = F.softmax(attack_model(y_pred)).cpu().detach()
        loss, confidence = CW_loss(pred_logit, y_label)
        adv_loss = - loss.view(N,1)
        A = (adv_loss - torch.mean(adv_loss))/(torch.std(adv_loss) + 1e-7)
        margin_loss = torch.mean(adv_loss)

        if iter%10 == 0:
            print ('Iter %d Adv LOSS %.4f Confidence %.4f' % (iter, margin_loss, confidence.numpy()))

        #update the latent variable
        if args.type == 'R':
            S_loss = torch.matmul(torch.transpose(pertur_s.view(N, -1), 0, 1), A).view(mu1.shape)
            s_context = s_context + lr/(N*S1)*S_loss.to(device)
        if args.type == 'Z':
            NES_loss = torch.matmul(torch.transpose(pertur.view(N, -1), 0, 1), A).view(mu1.shape)
            mu1 = mu1 + lr / (N * S1) * NES_loss.to(device)
        if args.type == 'RZ':
            S_loss = torch.matmul(torch.transpose(pertur_s.view(N, -1), 0, 1), A).view(mu1.shape)
            s_context = s_context + lr / (N * S1) * S_loss.to(device)
            NES_loss = torch.matmul(torch.transpose(pertur.view(N, -1), 0, 1), A).view(mu1.shape)
            mu1 = mu1 + lr / (N * S1) * NES_loss.to(device)

    y_hat = model.decoder(s_context, z_op, x_target)[0]
    if args.dataset == 'cifar-10':
        y_hat = y_hat.view(-1, channel_dim, channel_dim, c).permute(0, 3, 1, 2)
    perturbation = torch.clamp(y_hat - x_input, -eps, eps).cpu()
    return perturbation, False, 0, y_label.item()

def CW_loss(pred, y_label):
    pred_log = torch.log(pred)
    target_onehot = torch.zeros((1, 10))
    target_onehot[0][y_label] = 1
    y_pred = torch.sum(target_onehot*pred_log, dim=1)
    other = torch.max((1. - target_onehot) * pred_log - target_onehot * 10000., dim=1)[0]
    # print (other.shape)
    confidence = torch.mean(torch.sum(target_onehot*pred, dim=1))
    loss = y_pred - other
    # loss = torch.clamp(y_pred - other, 0., 1000)
    return loss, confidence

def load_attack_model(dataset):
    if dataset == 'mnist':
        model = mnist_mlp.mnist(pretrained=True, model_path=args.attack_path)
        model = model.cuda()
        model = model.eval()
        print ('Load MNIST Model')
    if dataset == 'cifar-10':
        if args.model == 'wrn':
            config = {
                "arch": "wrn",
                "base_channels": 16,
                "depth": 16,
                "widening_factor": 8,
                "drop_rate": 0,
                "n_classes": 10,
                "input_shape": (args.batch_size, 3, 32, 32),
            }
            print ('WRN Model')
            model = torch.nn.DataParallel(WRN(config))
            # print (torch.load(args.attack_path))
            model.load_state_dict(torch.load(args.attack_path))
            model = model.cuda()
            model = model.eval()
    return model

def load_NP_model(NP_path, A_path):
    # NP_model = NP(args).to(device)
    if args.dataset == 'mnist':
        NP_model = torch.load(NP_path)
        A_Matrix = None
        if A_path != None:
            A_Matrix = torch.load(A_path)
        NP_model.training = False
    if args.dataset == 'cifar-10':
        NP_model = torch.load(NP_path)
        NP_model.training = False
        A_Matrix = None
    print ('Load NP Model')
    return NP_model, A_Matrix

if args.dataset == 'mnist':
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],
                                                                                 std=[1])])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transform_),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    channel_dim, data_dim = 28, 784
    c = 1
    from ANP.ANP_MNIST import NP

elif args.dataset == 'cifar-10':
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_path, train=True, download=True,
                       transform=transform_),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_path, train=False, transform=transform_),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    channel_dim = 32
    c = 3
    data_dim = channel_dim * channel_dim * 3
    from ANP.ANP_CIFAR import NP

if args.A_path != 'no':
    NP_model, A_Matrix = load_NP_model(args.NP_path, args.A_path)
else:
    NP_model, A_Matrix = load_NP_model(args.NP_path, None)
attack_model = load_attack_model(args.dataset)

if args.dataset == 'mnist':
    x_grid = generate_grid(channel_dim, channel_dim)
if args.dataset == 'cifar-10':
    x_grid = generate_grid_cifar(channel_dim, channel_dim, c)


attack_success = 0.
Test_Image = 0.
attack_success_images = []
test_images = []
avg_iters = 0.
log_time = 1
cnt = 0
avg_L1 = 0.
avg_L2 = 0.
avg_Lp = 0.

context_idx = get_context_idx(data_dim, data_dim)

root_path = args.root_path
corrects = 0.
total = 0.
for batch_idx, (y_all, y_label) in enumerate(test_loader):
    pred = attack_model(y_all.cuda())
    y_pred = np.argmax(pred.data.cpu().numpy(), axis=1)
    print('label', y_label.item(), 'pred', y_pred)
    print('evaluating %d image'% (Test_Image))
    total += 1
    if y_pred.item() == y_label.item():
        corrects += 1
    # continue

    if y_pred.item() != y_label.item():
        continue
    Test_Image += 1
    batch_size = y_all.shape[0]
    if args.dataset == 'mnist':
        y_all = y_all.view(batch_size, -1, 1)
    elif args.dataset == 'cifar-10':
        y_all = y_all.permute(0, 2, 3, 1).to(device).view(batch_size, -1, 1)
    perturbation, attack, iters, pred = NP_Attack(NP_model, y_all, y_label, args.N, attack_model)
    if attack == True:
        attack_success += 1
        avg_iters += iters
        perturbation = perturbation.detach().numpy()
        avg_L1 += np.sum(np.abs(perturbation))
        avg_L2 += np.sum((perturbation) ** 2) ** .5
        avg_Lp += np.max(np.abs(perturbation))
        print ('Predict %d, Label %d'%(pred, y_label.item()))
        ASR = float(attack_success) / Test_Image
        AI = avg_iters / float(attack_success)
        log_msg = 'Attacking %d Images Attack Success Rate: %.4f Avg Iter: %.4f Avg Query: %d Avg L1 %.4f Avg L2 %.4f Avg Lp %.4f \n ' % \
                  (Test_Image, ASR, AI, AI * args.N, avg_L1 / attack_success, avg_L2 / attack_success,
                   avg_Lp / attack_success)
        print (log_msg)
        if os.path.exists(root_path) == False:
            os.mkdir(root_path)
        if attack_success == 1:
            with open('%s/log_%s_%.3f_%d_%.4f_%d_%s_%s.txt' % (root_path, args.dataset, args.E, args.N, args.LR, args.I, args.model, args.type), 'w') as f:
                f.write(log_msg)
        else:
            with open('%s/log_%s_%.3f_%d_%.4f_%d_%s_%s.txt' % (root_path, args.dataset, args.E, args.N, args.LR, args.I, args.model, args.type), 'a') as f:
                f.write(log_msg)
    if Test_Image > args.images_num:
        break
    if len(attack_success_images) > 5 and cnt < log_time:
        AS_IMAGE = torch.cat(attack_success_images, dim=0)
        ORI_IMAGE = torch.cat(test_images, dim=0)
        Exp_Name = '%s_%d_%d_%.2f_%s' % (args.dataset, args.I, args.N, args.E, args.type)
        if args.dataset == 'mnist':
            sample_image(AS_IMAGE, ORI_IMAGE, context_idx, '%s/%s' % (root_path, Exp_Name), cnt)
        else:
            sample_image_cifar(AS_IMAGE, ORI_IMAGE, context_idx, '%s/%s' % (root_path, Exp_Name), cnt)
        cnt += 1
        attack_success_images = []
        test_images = []

print(corrects, total)
print(corrects/total)
# assert False

ASR = float(attack_success)/Test_Image
AI = avg_iters/float(attack_success)
print ('Attack Success Rate: %.4f'%(ASR))
print ('Average Iteration: %.4f:'%(AI))
log_msg = 'Attacking %d Images Attack Success Rate: %.4f Avg Iter: %.4f Avg Query: %d Avg L1 %.4f Avg L2 %.4f Avg Lp %.4f \n ' % \
                  (Test_Image, ASR, AI, AI * args.N, avg_L1 / attack_success, avg_L2 / attack_success,
                   avg_Lp / attack_success)
with open('%s/log_%s_%.3f_%d_%.4f_%d_%s_%s.txt' % (root_path, args.dataset, args.E, args.N, args.LR, args.I, args.model, args.type), 'a') as f:
    f.write(log_msg)












