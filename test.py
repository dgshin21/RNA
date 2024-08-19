'''
Codes adapted from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/test.py
which uses Apache-2.0 license.
'''
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.SCOODBenchmarkDataset import SCOODDataset
from datasets.ImbalanceImageNet import LT_Dataset
from models.resnet import ResNet18
from models.resnet_imagenet import ResNet50
from torch.utils.data import Subset

from utils.utils import *
from utils.ltr_metrics import *
from utils.ood_metrics import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def val_cifar():
    '''
    Evaluate ID acc and OOD detection on CIFAR10/100
    '''
    model.eval()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    probs_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_scores_fn(model, images)
            probs = F.softmax(logits, dim=1)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            probs_list.append(probs.max(dim=1).values.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)

    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    # confidence distribution of correct samples:
    ood_score_list, sc_labels_list = [], []
    with torch.no_grad():
        for images, sc_labels in ood_loader:
            images, sc_labels = images.cuda(), sc_labels.cuda()
            logits, scores = get_scores_fn(model, images)
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
            sc_labels_list.append(sc_labels.detach().cpu().numpy())
    ood_scores = np.concatenate(ood_score_list, axis=0)
    sc_labels = np.concatenate(sc_labels_list, axis=0)

    # move some elements in ood_scores to in_scores:
    fake_ood_scores = ood_scores[sc_labels>=0]
    real_ood_scores = ood_scores[sc_labels<0]
    real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)

    auroc, aupr, fpr95 = get_measures(real_ood_scores, real_in_scores)

    print("AUROC: %.2f, AUPR: %.2f, FPR95: %.2f, ACC: %.2f, MANY: %.2f, MEDIUM: %.2f, FEW: %.2f" % (auroc*100, aupr*100, fpr95*100, test_acc*100, many_acc*100, median_acc*100, low_acc*100))


def val_imagenet():
    '''
    Evaluate ID acc and OOD detection on ImageNet
    '''
    model.eval()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    probs_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_scores_fn(model, images)
            probs = F.softmax(logits, dim=1)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            probs_list.append(probs.max(dim=1).values.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    # in_probs = np.concatenate(probs_list, axis=0)
    if args.dout == 'imagenet-10k':
        np.save(os.path.join(save_dir, 'in_scores.npy'), in_scores)
        np.save(os.path.join(args.ckpt_path, 'in_labels.npy'), in_labels)
        np.save(os.path.join(save_dir, 'in_preds.npy'), in_preds)
    many_acc, median_acc, low_acc = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=False)

    # confidence distribution of correct samples:
    ood_score_list = []
    with torch.no_grad():
        for images, _ in ood_loader:
            images = images.cuda()
            logits, scores = get_scores_fn(model, images)
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
    ood_scores = np.concatenate(ood_score_list, axis=0)
    if args.dout == 'imagenet-10k':
        np.save(os.path.join(save_dir, 'ood_scores.npy'), ood_scores)

    auroc, aupr, fpr95 = get_measures(ood_scores, in_scores)

    # print:
    print("AUROC: %.2f, AUPR: %.2f, FPR95: %.2f, ACC: %.2f, MANY: %.2f, MEDIUM: %.2f, FEW: %.2f" % (auroc*100, aupr*100, fpr95*100, test_acc*100, many_acc*100, median_acc*100, low_acc*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a CIFAR Classifier')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='../data', help='Where you save all your datasets.')
    parser.add_argument('--dout', default='svhn', choices=['svhn', 'places365', 'cifar', 'texture', 'tin', 'lsun'], help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--test_batch_size', '--tb', type=int, default=256)
    parser.add_argument('--metric', default='msp', help='OOD detection metric')
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--ckpt', default='latest', choices=['latest', 'epoch'])
    parser.add_argument('--feature_dim', default=512, type=int)
    args = parser.parse_args()
    # print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    save_dir = os.path.join(args.ckpt_path)
    # print(save_dir)
    # create_dir(save_dir)

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
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
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt',
            transform=train_transform,
            subset_class_idx=np.arange(0, num_classes))
        test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_test.txt',
            transform=test_transform,
            subset_class_idx=np.arange(0, num_classes))
    else:
        raise NotImplementedError()
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)
    din_str = 'Din is %s with %d images' % (args.dataset, len(test_set))
    print(din_str)

    if args.dout == 'cifar':
        if args.dataset == 'cifar10':
            args.dout = 'cifar100'
        elif args.dataset == 'cifar100':
            args.dout = 'cifar10'
        else:
            raise NotImplementedError()
    if args.dataset in ['cifar10', 'cifar100']:
        ood_set = SCOODDataset(os.path.join(args.data_root_path, 'SCOOD'), id_name=args.dataset, ood_name=args.dout, transform=test_transform)
    elif args.dataset == 'imagenet':
        ood_set = ImageFolder(os.path.join(args.data_root_path, 'imagenet10k_ood_test'), transform=test_transform,
                              loader=pil_loader)
        class_count = torch.zeros(len(ood_set.class_to_idx))
        ood_subset_idx = []
        for idx, label in enumerate(ood_set.targets):
            if class_count.sum() == 50000:
                break
            if class_count[label] >= 50:
                continue
            class_count[label] += 1
            ood_subset_idx.append(idx)
        ood_set = Subset(ood_set, ood_subset_idx)
    else:
        raise NotImplementedError()
    ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, pin_memory=True)
    dout_str = 'Dout is %s with %d images' % (args.dout, len(ood_set))
    print(dout_str)
    img_num_per_cls = np.array(train_set.img_num_per_cls)

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes, feature_dim=args.feature_dim).cuda()
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes).cuda()
    else:
        raise NotImplementedError()


    # load model:
    ckpt = torch.load(os.path.join(args.ckpt_path, 'latest.pth'), map_location="cuda:0")['model']
    model.load_state_dict(ckpt, strict=False)
    model.requires_grad_(False)

    # select a detection function:
    if args.metric == 'msp':
        get_scores_fn = get_msp_scores
    elif args.metric == 'rep_norm':
        get_scores_fn = get_rep_norm_scores
    elif args.metric == 'energy':
        get_scores_fn = get_energy_scores
    else:
        raise NotImplementedError("The score metric is NOT IMPLEMENTED!")

    if args.dataset in ['cifar10', 'cifar100']:
        val_cifar()
    elif args.dataset == 'imagenet':
        val_imagenet()