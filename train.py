import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision import datasets

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.ImbalanceImageNet import LT_Dataset
from datasets.tinyimages_300k import TinyImages
from models.resnet import ResNet18
from models.resnet_imagenet import ResNet50

from utils.utils import *
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', '--cpus', type=int, default=0, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='../data', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet50'],
                        help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay_epochs', '--de', default=[60, 80], nargs='+', type=int,
                        help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--Lambda', default=0.5, type=float, help='RNA loss term balancing hyper-parameter')
    parser.add_argument('--num_ood_samples', default=300000, type=float, help='Number of OOD samples to use.')
    parser.add_argument('--tau', type=float, default=1, help='logit adjustment hyper-parameter')
    parser.add_argument('--suffix', default='', type=str, help='suffix after exp str')
    parser.add_argument('--save_root_path', '--srp', default='logs', help='save root path')
    parser.add_argument('--eval_period', default=10, type=int)
    # ddp
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'],
                        help='If true, use distributed data parallel')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    args = parser.parse_args()

    return args


def create_save_path():
    # mkdirs:
    decay_str = args.decay
    if args.decay == 'multisteps':
        decay_str += '-'.join(map(str, args.decay_epochs))
    opt_str = args.opt
    if args.opt == 'sgd':
        opt_str += '-m%s' % args.momentum
    opt_str = 'e%d-b%d-%s-lr%s-wd%s-%s' % (args.epochs, args.batch_size, opt_str, args.lr, args.wd, decay_str)
    exp_str = '%s' % (opt_str)
    if args.suffix:
        exp_str += '_%s' % args.suffix

    dataset_str = '%s-%s-OOD%d' % (
    args.dataset, args.imbalance_ratio, args.num_ood_samples) if 'imagenet' not in args.dataset else '%s-lt' % (
    args.dataset)
    save_dir = os.path.join(args.save_root_path, dataset_str, args.model, exp_str)
    create_dir(save_dir)
    print('Saving to %s' % save_dir)

    return save_dir


def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(gpu_id, ngpus_per_node, args):
    save_dir = args.save_dir

    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True
    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise NotImplementedError()

    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform,
                                     imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio,
                                    root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform,
                                      imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio,
                                     root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt',
            transform=train_transform,
            subset_class_idx=np.arange(0, num_classes))
        test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_val.txt',
            transform=test_transform,
            subset_class_idx=np.arange(0, num_classes))
    else:
        raise NotImplementedError()

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                              drop_last=True, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers,
                             drop_last=False, pin_memory=True)

    if args.dataset in ['cifar10', 'cifar100']:
        ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))
    elif args.dataset == 'imagenet':
        ood_set = datasets.ImageFolder(os.path.join(args.data_root_path, 'imagenet10k_extra_ood'),
                                       transform=train_transform, loader=pil_loader)
    else:
        raise NotImplementedError()

    if args.ddp:
        ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set)
    else:
        ood_sampler = None
    ood_loader = DataLoader(ood_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                            drop_last=True, pin_memory=True, sampler=ood_sampler)
    print('Training on %s with %d images and %d validation images | %d OOD training images.' % (
    args.dataset, len(train_set), len(test_set), len(ood_set)))

    # get prior distributions:
    img_num_per_cls = np.array(train_set.img_num_per_cls)
    prior = img_num_per_cls / np.sum(img_num_per_cls)
    prior = torch.from_numpy(prior).float().to(device)

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes, return_features=True, feature_dim=args.feature_dim).to(device)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes, return_features=True).to(device)
    else:
        raise NotImplementedError()
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False)

    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum,
                                    nesterov=True)
    else:
        raise NotImplementedError()
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.decay == 'multisteps':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)
    else:
        raise NotImplementedError()

    # train:
    training_losses, test_clean_losses = [], []
    f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []

    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')
    for epoch in range(args.epochs):
        if args.ddp:
            # reset sampler when using ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        training_loss_meter = AverageMeter()
        training_loss_LA_meter = AverageMeter()
        training_loss_RNA_meter = AverageMeter()
        current_lr = scheduler.get_last_lr()

        for batch_idx, ((in_data, labels), (ood_data, _)) in enumerate(zip(train_loader, ood_loader)):
            in_data, labels = in_data.to(device), labels.to(device)
            ood_data = ood_data.to(device)
            N_in = len(labels)

            all_data = torch.cat([in_data, ood_data], dim=0)  # shape=(Nin+Nout,C,W,H)

            # forward:
            all_logits, all_reps = model(all_data)
            in_logits = all_logits[:N_in]
            in_reps = all_reps[:N_in]
            adjusted_in_logits = in_logits + args.tau * prior.log()[None, :]
            LA_loss = F.cross_entropy(adjusted_in_logits, labels)

            if args.ddp:
                in_h = model.module.forward_projection(in_reps)
            else:
                in_h = model.forward_projection(in_reps)

            in_h_norm = in_h.norm(dim=1)
            RNA_loss = - torch.log(1 + in_h_norm).mean()
            loss = LA_loss + args.Lambda * RNA_loss

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            training_loss_LA_meter.append(LA_loss.item())
            training_loss_RNA_meter.append(RNA_loss.item())

            if batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss %.4f (%.4f, %.4f) | lr %s' % (
                    epoch, batch_idx, loss.item(), LA_loss.item(), RNA_loss.item(), current_lr)
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        # lr update:
        scheduler.step()

        if (epoch+1) % args.eval_period == 0:
            # eval on clean set:
            model.eval()
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            preds_list, labels_list = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits, _ = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)
                    loss = F.cross_entropy(logits, labels)
                    test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                    test_loss_meter.append(loss.item())
                    preds_list.append(pred)
                    labels_list.append(labels)

            preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
            labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()

            overall_acc = (preds == labels).sum().item() / len(labels)
            test_clean_losses.append(test_loss_meter.avg)
            overall_accs.append(overall_acc)

            val_str = 'epoch %d (test): ACC %.4f ' % (epoch, overall_acc)
            print(val_str)
            fp_val.write(val_str + '\n')
            fp_val.flush()

        # save pth:
        if args.ddp:
            torch.save({
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'training_losses': training_losses,
                'test_clean_losses': test_clean_losses,
                'f1s': f1s,
                'overall_accs': overall_accs,
                'many_accs': many_accs,
                'median_accs': median_accs,
                'low_accs': low_accs,
            },
                os.path.join(save_dir, 'latest.pth'))
        else:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'training_losses': training_losses,
                'test_clean_losses': test_clean_losses,
                'f1s': f1s,
                'overall_accs': overall_accs,
                'many_accs': many_accs,
                'median_accs': median_accs,
                'low_accs': low_accs,
            },
                os.path.join(save_dir, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()


if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_dir = create_save_path()
    args.save_dir = save_dir

    # set CUDA:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node, args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)
