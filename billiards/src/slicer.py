from PIL import Image
import numpy as np
import Augmentor
import shutil
import pickle
import os
import re

from . import log, utils


class Slicer:
    def __init__(self, config, label, selected_energy_levels, loglevel=log.NO):
        """Slicing of every label which corresponds to `label` (e.g. for
        'sinai' the following list of labels will be sliced: ['sinai_0', 'sinai_0.1', ...]).

        Args:
            config (Config):
            label (str): E.g. 'sinai'.
            selected_energy_levels (list):
            loglevel (int):

        """
        self.LOGLEVEL = loglevel
        self.config = config
        self.SELECTED_ENERGY_LEVELS = selected_energy_levels
        # Set `GIVEN_LABELS`.
        prefixed = label + '_'
        self.GIVEN_LABELS = list(filter(lambda x: x.startswith(prefixed), utils.listdir(self.config.RAW_PATH)))

    def __prepared_path_label(self, label):
        return os.path.join(self.config.RAW_PATH, label, self.config.PREPARED_FOLDER)

    def __pieces_path_label(self, label):
        return os.path.join(self.config.RAW_PATH, label, self.config.PIECES_FOLDER)

    def _slice_label(self, label):
        pieces_path = self.__pieces_path_label(label)
        prepared_path = self.__prepared_path_label(label)

        if self.REMOVE_OLD_SLICED:
            shutil.rmtree(pieces_path, ignore_errors=True)

        os.makedirs(pieces_path, exist_ok=True)
        os.makedirs(prepared_path, exist_ok = True)
        image_names = utils.listdir(prepared_path)
        image_names = filter(lambda x: utils.energy_level(x) in self.SELECTED_ENERGY_LEVELS, image_names)

        for image_name in image_names:
            log.debug(self.LOGLEVEL, 'Start with slicing {} of {}'.format(image_name, label))
            image_path = os.path.join(prepared_path, image_name)
            self._slice_image(image_path, pieces_path)

    def _slice_image(self, image_path, pieces_path):
        extension = image_path.split('.')[-1]
        img = Image.open(image_path)
        piece_num = 0

        while piece_num < self.config.NUM_PIECES:
            image_name = os.path.split(image_path)[-1].split('.')[0]
            piece_name = "{}_{}.{}".format(image_name, str(piece_num), extension)
            piece_path = os.path.join(pieces_path, piece_name)

            if self.REMOVE_OLD_SLICED or os.path.exists(piece_path) is False:
                working_piece = utils.random_piece(img, self.config.PIECE_WIDTH, self.config.PIECE_HEIGHT)
                working_piece = working_piece.resize((self.config.RESIZED_WIDTH, self.config.RESIZED_HEIGHT), Image.ANTIALIAS)
                clrs = working_piece.getcolors(3000)

                if clrs is None or utils.GREEN in list(map(lambda x: x[1], clrs)):
                    log.trash(self.LOGLEVEL, 'Missed')
                else:
                    working_piece = working_piece.convert('L')
                    working_piece.save(piece_path)
                    piece_num += 1
                    log.trash(self.LOGLEVEL, 'Success')

            elif self.REMOVE_OLD_SLICED is False and os.path.exists(piece_path):
                piece_num += 1

    def _dump_label(self, label):
        label_path = os.path.join(self.config.RAW_PATH, label)
        dump_images_path = os.path.join(label_path, 'images.pkl')
        dump_labels_path = os.path.join(label_path, 'labels.pkl')
        dump_energy_levels_path = os.path.join(label_path, 'energy_levels.pkl')

        if self.REMOVE_OLD_DUMPED or os.path.exists(dump_images_path) is False or \
           os.path.exists(dump_labels_path) is False:
            images_array, labels_array = self._load_images_labels(label_path)
            labels_array = np.array(labels_array)
            #####
            # Divide each element by maximum value.
            #
            images_array = np.array(images_array)
            # images_array = images_array.reshape((-1, self.config.RESIZED_WIDTH * self.config.RESIZED_HEIGHT))
            # maximums = np.max(images_array, axis=1)
            # images_array = np.divide(images_array, maximums.reshape((len(maximums), 1)))
            # images_array = images_array.reshape((-1, self.config.RESIZED_WIDTH, self.config.RESIZED_HEIGHT))
            #####

            if self.REMOVE_OLD_DUMPED or os.path.exists(dump_images_path) is False:
                with open(dump_images_path, 'wb') as f:
                    pickle.dump(images_array, f)

            if self.REMOVE_OLD_DUMPED or os.path.exists(dump_labels_path) is False:
                with open(dump_labels_path, 'wb') as f:
                    pickle.dump(labels_array, f)

        if self.REMOVE_OLD_DUMPED or os.path.exists(dump_energy_levels_path) is False:
            energy_levels = self._load_energy_levels(label_path)
            energy_levels = np.array(energy_levels)
            with open(dump_energy_levels_path, 'wb') as f:
                pickle.dump(energy_levels, f)

    def _load_images_labels(self, label_path):
        images_array = list()
        augment_path = os.path.join(label_path, self.config.AUGMENT_FOLDER)
        slices = utils.listdir(augment_path, ignore_folders=True)

        for slice_name in slices:
            img = Image.open(os.path.join(augment_path, slice_name)).convert('L')
            data = np.asarray(img, dtype="int32")
            images_array.append(data)

        
        label = os.path.split(label_path)[-1]
        labels_array = [label] * len(slices)
        log.debug(self.LOGLEVEL, '> labels_array: {}'.format(labels_array))
        return images_array, labels_array

    def _load_energy_levels(self, label_path):
        energy_levels = list()
        augment_path = os.path.join(label_path, self.config.AUGMENT_FOLDER)
        slices = utils.listdir(augment_path)

        for slice_name in slices:
            energy_levels.append(utils.energy_level(slice_name))

        return energy_levels

    def _augment_label(self, label):
        label_path = os.path.join(self.config.RAW_PATH, label)
        pieces_path = os.path.join(label_path, self.config.PIECES_FOLDER)
        augment_path = os.path.join(label_path, self.config.AUGMENT_FOLDER)
        output_path = os.path.join(pieces_path, self.config.OUTPUT_FOLDER)

        if self.REMOVE_OLD_AUGMENTED:
            shutil.rmtree(output_path, ignore_errors=True)
            shutil.rmtree(augment_path, ignore_errors=True)

        p = Augmentor.Pipeline(source_directory=pieces_path)
        p.rotate(probability=0.9, max_left_rotation=25, max_right_rotation=25)
        p.rotate90(probability=0.5)
        p.rotate270(probability=0.5)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.sample(self.config.NUM_AUGMENT)

        shutil.move(output_path, augment_path)

    def augment(self, remove_old_augmented=False):
        self.REMOVE_OLD_AUGMENTED = remove_old_augmented
        log.info(self.LOGLEVEL, '=======\nAugmentation\n=======')

        for label in self.GIVEN_LABELS:
            log.info(self.LOGLEVEL, 'Start with {}'.format(label))
            self._augment_label(label)

    def dump(self, remove_old_dumped=False):
        self.REMOVE_OLD_DUMPED = remove_old_dumped
        log.info(self.LOGLEVEL, '=======\nDumping\n=======')

        for label in self.GIVEN_LABELS:
            log.info(self.LOGLEVEL, 'Start with {}'.format(label))
            self._dump_label(label)

    def slice(self, remove_old_sliced=False):
        self.REMOVE_OLD_SLICED = remove_old_sliced
        log.info(self.LOGLEVEL, '=======\nSlicing\n=======')
        log.info(self.LOGLEVEL, "{} x {} --> {} x {}".format(self.config.PIECE_WIDTH, self.config.PIECE_HEIGHT,
                 self.config.RESIZED_WIDTH, self.config.RESIZED_HEIGHT))

        for label in self.GIVEN_LABELS:
            log.info(self.LOGLEVEL, 'Start with {}'.format(label))
            self._slice_label(label)


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
