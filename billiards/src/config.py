import os
import pickle
import itertools

import log, utils


class Config:
    CLASSES_MAPPING = {
        'chaotic': 0,
        'regular': 1
    }

    ROOT_PATH = 'data'

    RAW_FOLDER = 'raw'
    EXP_FOLDER = 'experiments'
    PKL_FOLDER = 'pkl'
    ORIGINAL_FOLDER = 'original'
    PIECES_FOLDER = 'pieces'
    AUGMENT_FOLDER = 'augmented'
    PREPARED_FOLDER = 'prepared'
    OUTPUT_FOLDER = 'output'
    ENERGY_LEVELS_FOLDER = 'energy_levels'

    INFO_FILE = 'config.txt'
    INFO_PKL = 'config.pkl'

    PIECE_WIDTH = 350
    PIECE_HEIGHT = 350
    NUM_PIECES = 100
    DOWNSAMPLING_FACTOR = 10
    TEST_SIZE = 0.33

    LOGLEVEL = log.INFO

    NUM_AUGMENT = 2000

    def __init__(self):
        """Set values of computed attributes.

        """
        self.RESIZED_WIDTH = self.PIECE_WIDTH // self.DOWNSAMPLING_FACTOR
        self.RESIZED_HEIGHT = self.PIECE_HEIGHT // self.DOWNSAMPLING_FACTOR

        self.ROOT_PATH = os.path.expanduser(self.ROOT_PATH)
        self.RAW_PATH = os.path.join(self.ROOT_PATH, self.RAW_FOLDER)

        self.BOUNDING_LABELS = list(itertools.chain(*self.SPLITTING_INFO.values()))
        self.NAME = self._name()
        self.GIVEN_LABELS = self._given_labels()

    def __str__(self):
        to_str = list()
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                to_str.append("{:30} {}".format(a, getattr(self, a)))
        to_str.append("\n")
        return '\n'.join(to_str)

    def _name(self):
        name = ''
        for key in self.SPLITTING_INFO:
            for label in self.SPLITTING_INFO[key]:
                name += label
                name += ', '
            name = name[:-2]
            name += ' vs. '
        name = name[:-4]
        return name

    def _given_labels(self):
        bounding_labels = list(itertools.chain(*self.SPLITTING_INFO.values()))
        prefixes = set(map(lambda x: x.split('_')[0] + '_', bounding_labels))
        given_labels = list(map(lambda prefix: list(filter(lambda x: x.startswith(prefix),
                                utils.listdir(self.RAW_PATH))), prefixes))
        given_labels = list(itertools.chain(*given_labels))
        return given_labels

    def display(self):
        """Display Configuration values.

        """
        print(self.__str__())


def load_config(exp_folder_path):
    """Load saved config from folder `exp_folder_path`.

    Args:
        exp_folder_path (str):

    Returns:
        Config:

    """

    with open(os.path.join(exp_folder_path, Config.INFO_PKL), 'rb') as f:
        config = pickle.load(f)

    return config
