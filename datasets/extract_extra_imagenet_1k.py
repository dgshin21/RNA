import os, argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--data_root_path', '--drp', default='../data', help='data root path')
args = parser.parse_args()

with open("imagenet_extra_1k_wnid_list.txt", "r") as fp:
    imagenet_extra_1k_wnid_list = fp.read().splitlines()

imagenet10k_dir = os.path.join(args.data_root_path, 'imagenet10k')
imagenet10k_dir_extra_ood = os.path.join(args.data_root_path, 'imagenet10k_extra_ood')

for item in os.listdir(imagenet10k_dir):
    wnid = item.split('.')[0]
    if wnid in imagenet_extra_1k_wnid_list:
        if os.path.exists(os.path.join(imagenet10k_dir_extra_ood, wnid)):
            continue
        else:
            os.makedirs(os.path.join(imagenet10k_dir_extra_ood, wnid))
            os.system('tar xvf %s -C %s' % (os.path.join(imagenet10k_dir, item), os.path.join(imagenet10k_dir_extra_ood, wnid)))


