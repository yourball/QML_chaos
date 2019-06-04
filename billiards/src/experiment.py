from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import pickle
import torch
import os

from config import load_config, Config
from myconfig import MyConfig

import log, utils, billiard




class Experiment:
    def __init__(self, expn=None, exp_path=None, config=None, loglevel=log.NO):
        self.LOGLEVEL = loglevel

        myconfig = MyConfig()
        if expn is not None and exp_path is not None:
            self.EXPN = expn
            self.FOLDER = os.path.join(exp_path, str(expn))
            self.config = myconfig #load_config(self.FOLDER)
            log.info(self.LOGLEVEL, "Load experiment #{}".format(self.EXPN))
        elif config is not None:
            self.config = config
            exp_path = os.path.join(self.config.ROOT_PATH, self.config.EXP_FOLDER)
            os.makedirs(exp_path, exist_ok=True)
            self.EXPN = 1 + len(utils.listdir(exp_path))
            self.FOLDER = os.path.join(self.config.ROOT_PATH, self.config.EXP_FOLDER, str(self.EXPN))
            log.info(self.LOGLEVEL, self.__str__())
        else:
            raise RuntimeError()

    def __repr__(self):
        return self.config.__str__()

    def __prepared_path_label(self, label):
        return os.path.join(self.config.RAW_PATH, label, self.config.PREPARED_FOLDER)

    def __original_path_label(self, label):
        return os.path.join(self.config.RAW_PATH, label, self.config.ORIGINAL_FOLDER)

    def _log_file(self):
        os.makedirs(self.FOLDER, exist_ok=True)
        with open(os.path.join(self.FOLDER, self.config.INFO_FILE), 'w') as f:
            f.write(self.__repr__())
        with open(os.path.join(self.FOLDER, self.config.INFO_PKL), 'wb') as f:
            pickle.dump(self.config, f)

    def _binarize_labels(self, labels):
        binarization_mapping = dict()

        for key in self.config.SPLITTING_INFO:
            for label in self.config.SPLITTING_INFO[key]:
                binarization_mapping[label] = self.config.CLASSES_MAPPING[key]

        log.debug(self.LOGLEVEL, '> binarization_mapping: {}'.format(binarization_mapping))
        log.debug(self.LOGLEVEL, '> SPLITTING_INFO: {}'.format(self.config.SPLITTING_INFO))
        binary_labels = np.array(list(map(lambda x: binarization_mapping[x], labels)))
        return binary_labels

    def normalize(self, loaded_images):
        log.debug(self.LOGLEVEL, '> normalizing')
        norm_images = loaded_images
        for (indx, image) in enumerate(loaded_images):
            norm_images[indx] = image/np.sum(np.sum(image**2))
        return norm_images

    def _load_dataset(self):
        loaded_images = list()
        loaded_labels = list()
        loaded_energy_levels = list()

        for label in self.config.BOUNDING_LABELS:
            label_path = os.path.join(self.config.RAW_PATH, label)
            dump_images_path = os.path.join(label_path, 'images.pkl')
            dump_labels_path = os.path.join(label_path, 'labels.pkl')
            dump_energy_levels_path = os.path.join(label_path, 'energy_levels.pkl')
            with open(dump_images_path, 'rb') as f:
                tmp_images = pickle.load(f)
            with open(dump_labels_path, 'rb') as f:
                tmp_labels = pickle.load(f)
            with open(dump_energy_levels_path, 'rb') as f:
                tmp_energy_levels = pickle.load(f)
            loaded_images.extend(tmp_images)
            loaded_labels.extend(tmp_labels)
            loaded_energy_levels.extend(tmp_energy_levels)

        loaded_images = np.array(loaded_images)
        loaded_labels = np.array(loaded_labels)
        loaded_energy_levels = np.array(loaded_energy_levels)

        #loaded_images = self.normalize(loaded_images)
        return loaded_images, loaded_labels, loaded_energy_levels

    def _dump(self, dump_mapping):
        os.makedirs(os.path.join(self.FOLDER, self.config.PKL_FOLDER), exist_ok=True )

        for pkl_file in dump_mapping:
            log.info(self.LOGLEVEL, 'Start with {}'.format(pkl_file))
            pkl_path = os.path.join(self.FOLDER, self.config.PKL_FOLDER, pkl_file)
            data_array = dump_mapping[pkl_file]
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_array, f)

        self._log_file()

    def prepare_validation(self):
        log.info(self.LOGLEVEL, '=======\nValidation\n=======')
        loaded_images, loaded_labels, loaded_energy_levels = self._load_dataset()
        log.debug(self.LOGLEVEL, '> loaded_labels: {}'.format(loaded_labels))
        loaded_labels = self._binarize_labels(loaded_labels)

        sizes = str(len(loaded_images))
        self.config.SIZES = sizes
        sizes_str = 'Sizes: {}'.format(sizes)
        log.info(self.LOGLEVEL, sizes_str)

        dump_mapping = {
            'validation-images.pkl': loaded_images,
            'validation-labels.pkl': loaded_labels,
            'validation-energy-levels.pkl': loaded_energy_levels
        }
        self._dump(dump_mapping)

    def prepare_train_test(self):
        log.info(self.LOGLEVEL, '=======\nTrain/Test Split\n=======')
        loaded_images, loaded_labels, loaded_energy_levels = self._load_dataset()

        loaded_labels = self._binarize_labels(loaded_labels)
        idxs = np.arange(len(loaded_labels))
        idxs_train, idxs_test = train_test_split(idxs, test_size=self.config.TEST_SIZE,
                                                 random_state=42)
        X_train = loaded_images[idxs_train]
        X_test = loaded_images[idxs_test]
        y_train = loaded_labels[idxs_train]
        y_test = loaded_labels[idxs_test]
        energy_levels_train = loaded_energy_levels[idxs_train]
        energy_levels_test = loaded_energy_levels[idxs_test]

        sizes = ' '.join(map(lambda x: str(len(x)), [X_train, X_test]))
        self.config.SIZES = sizes
        sizes_str = 'Sizes: {}'.format(sizes)
        log.info(self.LOGLEVEL, sizes_str)

        dump_mapping = {
            'train-images.pkl': X_train,
            'test-images.pkl': X_test,

            'train-labels.pkl': y_train,
            'test-labels.pkl': y_test,

            'train-energy-levels.pkl': energy_levels_train,
            'test-energy-levels.pkl': energy_levels_test
        }
        self._dump(dump_mapping)
        return (X_train, y_train, X_test, y_test)

    def get_train_test_loaders(self, batch_size, test_batch_size, selected_energy_levels):
        train_billiard = billiard.Billiard(self.FOLDER, mode='train', raw_folder='pkl', processed_folder='pt',
                                           loglevel=self.LOGLEVEL, remove_old=True,
                                           selected_energy_levels=selected_energy_levels)
        test_billiard = billiard.Billiard(self.FOLDER, mode='test', raw_folder='pkl', processed_folder='pt',
                                          loglevel=self.LOGLEVEL, remove_old=True,
                                          selected_energy_levels=selected_energy_levels)

        train_loader = torch.utils.data.DataLoader(train_billiard, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_billiard, batch_size=test_batch_size, shuffle=True)
        return test_loader, train_loader

    def get_validation_loader(self, batch_size, selected_energy_levels):
        validation_billiard = billiard.Billiard(self.FOLDER, mode='validation', raw_folder='pkl', processed_folder='pt',
                                                loglevel=self.LOGLEVEL, remove_old=True,
                                                selected_energy_levels=selected_energy_levels)
        validation_loader = torch.utils.data.DataLoader(validation_billiard, batch_size=batch_size, shuffle=True)
        return validation_loader

    def remove(self):
        if os.path.exists(self.FOLDER):
            shutil.rmtree(self.FOLDER, ignore_errors=True)


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
