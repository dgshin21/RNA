import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', '--drp', default='../datasets', help='data root path')
args = parser.parse_args()

with open("imagenet_ood_test_1k_wnid_list.txt", "r") as fp:
    imagenet_ood_test_1k_wnid_list = fp.read().splitlines()

imagenet10k_dir = os.path.join(args.data_root_path, 'imagenet10k')
imagenet10k_dir_ood_test = os.path.join(args.data_root_path, 'imagenet10k_ood_test')
for item in os.listdir(imagenet10k_dir):
    wnid = item.split('.')[0]
    if wnid in imagenet_ood_test_1k_wnid_list:
        if os.path.exists(os.path.join(imagenet10k_dir_ood_test, wnid)):
            continue
        else:
            os.makedirs(os.path.join(imagenet10k_dir_ood_test, wnid))
            os.system('tar xvf %s -C %s' % (os.path.join(imagenet10k_dir, item), os.path.join(imagenet10k_dir_ood_test, wnid)))
