import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import numpy as np
import pickle
import shutil

import utils
import log


class Billiard(data.Dataset):
    """`Billiard` Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    validation_file = 'validation.pt'
    train_file = 'train.pt'
    test_file = 'test.pt'
    classes = ['1 - regular', '0 - chaotic']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        return self.data_labels

    def __init__(self, root, mode='train', logging_level=None, transform=transforms.Compose([transforms.ToTensor()]), target_transform=None,
                 process=True, raw_folder='raw', processed_folder='processed',
                 remove_old=False, selected_energy_levels=None, loglevel=log.NO):
        self.loglevel = loglevel

        self.root = os.path.expanduser(root)
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder

        self.mode = mode  # Train, test or validation set.

        self.transform = transform
        self.target_transform = target_transform
        self.remove_old = remove_old
        self.selected_energy_levels = selected_energy_levels

        if process:
            self._process()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use process=True to process it')

        self.data_images, self.data_labels, self.data_energy_levels = self._load_pt()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target, energy_level = self.data_images[index], self.data_labels[index], self.data_energy_levels[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image.

        img = img.numpy() / 255.
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, energy_level

    def __len__(self):
        return len(self.data_labels)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _load_pt(self):
        if self.mode == 'validation':
            return torch.load(os.path.join(self.root, self.processed_folder, self.validation_file))
        elif self.mode == 'train':
            return torch.load(os.path.join(self.root, self.processed_folder, self.train_file))
        elif self.mode == 'test':
            return torch.load(os.path.join(self.root, self.processed_folder, self.test_file))

    def _mask(self, energy_levels):
        mask = torch.Tensor(np.isin(energy_levels, self.selected_energy_levels).astype(int))
        mask = mask.type(torch.uint8)
        return mask

    def _check_exists(self):
        train_test_bool = os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
        validation_bool = os.path.exists(os.path.join(self.root, self.processed_folder, self.validation_file))
        return train_test_bool or validation_bool

    def _process(self):
        """Process the Billiard data if it doesn't exist in processed_folder already.

        """
        if self.remove_old:
            shutil.rmtree(os.path.join(self.root, self.processed_folder), ignore_errors=True)
        elif self._check_exists():
            return

        os.makedirs(os.path.join(self.root, self.raw_folder), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.processed_folder), exist_ok=True)

        # Process and save as torch files.
        log.info(self.loglevel, 'Processing...')

        if self.mode == 'validation':
            validation_images = read_file(os.path.join(self.root, self.raw_folder, 'validation-images.pkl'))
            validation_labels = read_file(os.path.join(self.root, self.raw_folder, 'validation-labels.pkl'))
            validation_energy_levels = read_file(os.path.join(self.root, self.raw_folder, 'validation-energy-levels.pkl'))

            if len(self.selected_energy_levels) > 0:
                validation_mask = self._mask(validation_energy_levels)
                validation_images = validation_images[validation_mask]
                validation_labels = validation_labels[validation_mask]
                validation_energy_levels = validation_energy_levels[validation_mask]

            validation_set = (
                validation_images,
                validation_labels,
                validation_energy_levels
            )
            # Dump as torch tensors.
            with open(os.path.join(self.root, self.processed_folder, self.validation_file), 'wb') as f:
                torch.save(validation_set, f)
        elif self.mode == 'train' or self.mode == 'test':
            train_images = read_file(os.path.join(self.root, self.raw_folder, 'train-images.pkl'))
            train_labels = read_file(os.path.join(self.root, self.raw_folder, 'train-labels.pkl'))
            train_energy_levels = read_file(os.path.join(self.root, self.raw_folder, 'train-energy-levels.pkl'))

            if len(self.selected_energy_levels) > 0:
                train_mask = self._mask(train_energy_levels)
                train_images = train_images[train_mask]
                train_labels = train_labels[train_mask]
                train_energy_levels = train_energy_levels[train_mask]

            test_images = read_file(os.path.join(self.root, self.raw_folder, 'test-images.pkl'))
            test_labels = read_file(os.path.join(self.root, self.raw_folder, 'test-labels.pkl'))
            test_energy_levels = read_file(os.path.join(self.root, self.raw_folder, 'test-energy-levels.pkl'))

            if len(self.selected_energy_levels) > 0:
                test_mask = self._mask(test_energy_levels)
                test_images = test_images[test_mask]
                test_labels = test_labels[test_mask]
                test_energy_levels = test_energy_levels[test_mask]

            train_set = (
                train_images,
                train_labels,
                train_energy_levels
            )
            test_set = (
                test_images,
                test_labels,
                test_energy_levels
            )
            # Dump as torch tensors.
            with open(os.path.join(self.root, self.processed_folder, self.train_file), 'wb') as f:
                torch.save(train_set, f)
            with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
                torch.save(test_set, f)

        log.info(self.loglevel, 'Done!')


def read_file(path):
    with open(path, 'rb') as f:
        parsed = pickle.load(f)
        return torch.from_numpy(parsed)


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
