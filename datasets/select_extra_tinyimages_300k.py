import os, argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', '--drp', default='../data', help='data root path')
args = parser.parse_args()

tinyimagenet_dir = os.path.join(args.data_root_path, 'TinyImages')
total_samples = 300000      # 300k
num_classes = 1239
num_imgs_per_class = int(np.floor(total_samples / num_classes))
num_one_more_classes = total_samples - num_classes * num_imgs_per_class
print(num_imgs_per_class, num_one_more_classes)

one_more_classes = np.random.choice(np.arange(num_classes), size=num_one_more_classes, replace=False)
# imgs = []
imgs = np.ndarray(shape=(total_samples, 32, 32, 3), dtype=np.uint8)
for i in range(num_classes):
    print('image collecting from class %d...' % i)
    img_dir = args.data_root_path + '/TinyImages/images_%d' % i
    if i in one_more_classes:   # num_imgs_per_class + 1
        num_samples = num_imgs_per_class + 1
    else:                       # num_imgs_per_class
        num_samples = num_imgs_per_class

    idx = np.random.choice(np.arange(1000), size=num_samples, replace=False)

    for j in range(num_samples):
        img_path = img_dir + '/img_%d' % (i * 1000 + idx[j]) + '.jpg'
        k = np.random.randint(64)
        img = Image.open(img_path).crop((0, k * 32, 32, (k + 1) * 32))
        img_array = np.array(img)
        imgs[i, :, :, :] = img

print(len(imgs), imgs[0].shape, imgs[-1].size)
np.save(args.data_root_path + '/TinyImages/300K_random_images.npy', imgs)
# np.save('./300K_random_images.npy', imgs)


