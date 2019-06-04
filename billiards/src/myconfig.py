import numpy as np
from config import Config


class MyConfig(Config):

    SPLITTING_INFO = {
     'chaotic': ['sinai_0.4'],
     'regular': ['sinai_0']
    }


    NUM_PIECES = 200
    NUM_AUGMENT = 4000

    PIECE_WIDTH = 250
    PIECE_HEIGHT = 250

    RESIZED_WIDTH = 40
    RESIZED_HEIGHT = 40
    SELECTED_ENERGY_LEVELS = np.arange(460, 500)
