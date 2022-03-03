# partition_food101.py
# Dan Popp
# 3/03/21
#
# This script will partition the food101 dataset into the testing and training split so it can be parsed
# as an ImageFolder dataset
import argparse
import os
import shutil
import sys

from tqdm import tqdm


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def partition_food101(path):
    dataset_dir = os.path.join(path, 'dataset')
    make_dir(dataset_dir)

    for split in ['train', 'test']:
        print('Paritioning %s split...' % split)

        split_file_path = os.path.join(path, 'meta', split + '.txt')
        split_dir = os.path.join(dataset_dir, split)
        make_dir(split_dir)
        with open(split_file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, file=sys.stdout):
                image_file = line.strip() + '.jpg'
                class_dir = os.path.dirname(image_file)
                make_dir(os.path.join(split_dir, class_dir))
                orig_file = os.path.join(path, 'images', image_file)
                dst_file = os.path.join(split_dir, image_file)

                shutil.copy(orig_file, dst_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='The file path to the CUB200 dataset directory')
    args = parser.parse_args()

    path = args.dataset_path

    partition_food101(path)


if __name__ == '__main__':
    main()
