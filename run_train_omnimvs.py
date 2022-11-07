# run_test_omnimvs.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import time
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
# Torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
#from torchvision.utils import make_grid
# Internal modules
from dataset import Dataset
from utils.common import *
from utils.image import *
from module.network import OmniMVSNet
from module.loss_functions import *

# Initialize
#GPU_ID = 0
#os.putenv('CUDA_VISIBLE_DEVICES', str(GPU_ID))
#torch.backends.cudnn.benchmark = True
#torch.backends.cuda.benchmark = True

opts = Edict()

# Dataset & sweep arguments
opts.train_dbname = 'dsta_train'
opts.val_dbname = 'dsta_val'
opts.db_root = './data'
opts.data_opts = Edict()
opts.data_opts.phi_deg = 45.0
opts.data_opts.num_invdepth = 192
opts.data_opts.equirect_size = (160, 640)
opts.net_opts = Edict()
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth

# Train
opts.train_opts = Edict()
opts.train_opts.epochs = 50
opts.train_opts.batch_size = 1
opts.train_opts.lr = 5e-4
opts.train_opts.momentum = 0.9
opts.train_opts.optim = 'SGD'
# Contd training
opts.train_opts.ckpt_path = './checkpoints/omnimvs_1103-2002/checkpoints_14.pth'
# opts.train_opts.ckpt_path = None

# Logs
opts.log_opts = Edict()
opts.log_opts.log_interval = 100
opts.log_opts.viz_log_interval = 250
# opts.log_opts.log_dir = os.path.join(
#     'checkpoints',
#     f'omnimvs_1103-2002'
# )
opts.log_opts.log_dir = os.path.join(
    'checkpoints',
    f'omnimvs_{datetime.now().strftime("%m%d-%H%M")}'
)

# Results
opts.vis = True
opts.save_result, opts.save_misc = True, True
opts.result_dir = osp.join('./results', opts.train_dbname)
opts.out_invdepth_fmt = osp.join(opts.result_dir, '%05d.tiff')
opts.out_entropy_fmt = osp.join(opts.result_dir, '%05d_entropy.tiff')
opts.out_misc_fmt = osp.join(opts.result_dir, '%05d.png')

if opts.vis:
    fig = plt.figure(frameon=False, figsize=(25,10), dpi=40)
    plt.ion()
    plt.show()

def run_epoch(net, data, grids, epoch, writer, optimizer=None, is_train=True):

    pbar = tqdm(range(data.data_size))
    errors = np.zeros((data.data_size, 6))

    loss_func = nn.SmoothL1Loss()

    shuffled_order = list(range(data.data_size))

    # Adding data shuffle while training.
    if is_train:
        shuffled_order = np.random.permutation(shuffled_order)

    for d in pbar:
        fidx = data.frame_idx[shuffled_order[d]]
        imgs, gt, valid, raw_imgs = data.loadSample(fidx)
        toc, toc2 = 0, 0
        # net.eval()
        tic = time.time()
        imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]

        # Running model.
        if is_train:
            invdepth_idx, prob, _ = net(imgs, grids, out_cost=True)
        else:
            with torch.no_grad():
                invdepth_idx, prob, _ = net(imgs, grids, out_cost=True)

        #invdepth_idx = toNumpy(invdepth_idx)
        invdepth = data.indexToInvdepth(invdepth_idx)
       # gt = toNumpy(gt)
        entropy = toNumpy(torch.sum(-torch.log(prob + EPS) * prob, 0))
        toc = time.time() - tic

        loss_l1 = loss_func(invdepth_idx, gt.reshape(invdepth_idx.shape).cuda())

        if is_train:
            optimizer.zero_grad()
            loss_l1.backward()
            optimizer.step()

        # update progress bar
        errors[d, 0] = loss_l1.item()
        errors[d, 1:] = data.evalError(invdepth_idx, gt, valid)
        display = OrderedDict(epoch=f"{epoch:>2}", loss=f"{loss_l1.item():.4f}")
        pbar.set_postfix(display)

        # log loss, metrics etc.
        niter = epoch * data.data_size + d
        if niter % opts.log_opts.log_interval == 0:
            writer.add_scalar(
                'train/loss' if is_train else 'val/loss',
                loss_l1.item(),
                niter
            )
            writer.add_scalar(
                'train/e1' if is_train else 'val/e1',
                errors[d, 1],
                niter
            )
            writer.add_scalar(
                'train/e3' if is_train else 'val/e3',
                errors[d, 2],
                niter
            )
            writer.add_scalar(
                'train/e5' if is_train else 'val/e5',
                errors[d, 3],
                niter
            )
            writer.add_scalar(
                'train/mae' if is_train else 'val/mae',
                errors[d, 4],
                niter
            )
            writer.add_scalar(
                'train/rms' if is_train else 'val/rms',
                errors[d, 5],
                niter
            )

        if niter % opts.log_opts.viz_log_interval == 0:
            vis_img = data.makeVisImage(
                raw_imgs,
                invdepth[0].detach().cpu(),
                entropy,
                gt.detach().cpu()
            )
            writer.add_image(
                'train/vis_img' if is_train else 'val/vis_img',
                np.transpose(vis_img, (2, 0, 1)),
                niter
            )
            writer.add_image(
                'train/pred' if is_train else 'val/pred',
                invdepth_idx / opts.data_opts.num_invdepth,
                niter
            )
            writer.add_image(
                'train/gt' if is_train else 'val/gt',
                gt.reshape(invdepth_idx.shape) / opts.data_opts.num_invdepth,
                niter
            )

        del invdepth, invdepth_idx, gt, raw_imgs, imgs

    return errors

def main():
    train_data = Dataset(opts.train_dbname, opts.data_opts, db_root=opts.db_root)
    val_data = Dataset(opts.val_dbname, opts.data_opts, db_root=opts.db_root)

    net = OmniMVSNet(opts.net_opts).cuda()
    start_epoch_i = 0

    train_grids = [torch.tensor(grid, requires_grad=False).cuda() \
        for grid in train_data.grids]
    val_grids = [torch.tensor(grid, requires_grad=False).cuda() \
        for grid in val_data.grids]

    if not osp.exists(opts.result_dir):
        os.makedirs(opts.result_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.result_dir))

    if opts.train_opts.optim == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=opts.train_opts.lr,
            momentum=opts.train_opts.momentum
        )
    else:
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=opts.train_opts.lr,
            momentum=opts.train_opts.momentum
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2 * opts.train_opts.epochs // 3,
        gamma=0.1
    )

    if opts.train_opts.ckpt_path is not None:
        snapshot = torch.load(opts.train_opts.ckpt_path)
        net.load_state_dict(snapshot['net_state_dict'])
        start_epoch_i = snapshot['epoch']
        optimizer.load_state_dict(snapshot['optimizer'])
        scheduler.load_state_dict(snapshot['scheduler'])

        if 'log_path' in snapshot.keys():
            opts.log_opts.log_dir = snapshot['log_path']

    # Creating a tb writer.
    writer = SummaryWriter(log_dir=opts.log_opts.log_dir)

    for i in range(start_epoch_i, start_epoch_i + opts.train_opts.epochs):
        print(f'Epoch {i}:')

        net.train()
        metrics = run_epoch(net, train_data, train_grids, i, writer, optimizer=optimizer)
        metrics = np.mean(metrics, axis=0)
        print(
            f'\tTrain Metrics - Avg. Loss {metrics[0]}, Avg. E1 {metrics[1]}, ' +
            f'Avg. E3 {metrics[2]}, Avg. E5 {metrics[3]}, ' +
            f'Avg. MAE {metrics[4]}, Avg. RMS {metrics[5]}'
        )

        net.eval()
        metrics = run_epoch(net, val_data, val_grids, i, writer, is_train=False)
        metrics = np.mean(metrics, axis=0)
        print(
            f'\tVal Metrics - Avg. Loss {metrics[0]}, Avg. E1 {metrics[1]}, ' +
            f'Avg. E3 {metrics[2]}, Avg. E5 {metrics[3]}, ' +
            f'Avg. MAE {metrics[4]}, Avg. RMS {metrics[5]}'
        )
        writer.add_scalar('val/loss_ave', metrics[0], i)

        scheduler.step()

        # save data here
        save_data = {
            'epoch': i + 1,
            'net_state_dict': net.state_dict(),
            'CH': net.opts.CH,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'log_path': opts.log_opts.log_dir,
        }
        torch.save(
            save_data,
            os.path.join(opts.log_opts.log_dir, f'checkpoints_{i}.pth')
        )

if __name__ == "__main__":
    main()
