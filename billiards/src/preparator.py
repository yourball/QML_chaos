import shutil
import cv2
import os

from . import utils, log


class Preparator:
    def __init__(self, config, label, loglevel=log.NO):
        """Performs preparaton for given `label`, i.e. preparation of every label
        corresponding to `label` (e.g. for 'sinai' the following list of labels
        will be prepared: ['sinai_0', 'sinai_0.1', ...]).

        Args:
            config (qchaos.config.Config):
            label (str): E.g. 'sinai'.
            loglevel (int):

        """
        self.LOGLEVEL = loglevel
        self.config = config

        # Set `GIVEN_LABELS`.
        prefixed = label + '_'
        self.GIVEN_LABELS = list(filter(lambda x: x.startswith(prefixed), utils.listdir(self.config.RAW_PATH)))

    def __prepared_path_label(self, label):
        return os.path.join(self.config.RAW_PATH, label, self.config.PREPARED_FOLDER)

    def __original_path_label(self, label):
        return os.path.join(self.config.RAW_PATH, label, self.config.ORIGINAL_FOLDER)

    def _prepare_image(self, image_path, minw=50, minh=50):
        img = cv2.imread(image_path)
        img_original = img.copy()

        def fail():
            log.info(self.config.LOGLEVEL, '> FAILED: {}'.format(image_path))
            return img_original

        cnts = utils.find_contours(img, minw=minw, minh=minh, debug=False)
        if len(cnts) > 0:
            result = utils.apply_max_contour(img_original, cnts, debug=False)
            clrs, width, height = utils.colors_from_array(result)

            if utils.GREEN in clrs:
                green_portion = clrs[utils.GREEN] / (width * height)
                if green_portion > 0.4:
                    return fail()

            return result
        else:
            return fail()

    def _prepare_label(self, label):
        prepared_path = self.__prepared_path_label(label)
        original_path = self.__original_path_label(label)

        if self.REMOVE_OLD_PREPARED:
            shutil.rmtree(prepared_path, ignore_errors=True)
        os.makedirs(prepared_path, exist_ok=True)
        image_names = utils.listdir(original_path)
        # filtering psi2_{}.png files
        image_names = list(filter(lambda x: x.startswith('psi2') is True, image_names))
        for image_name in image_names:
            image_path = os.path.join(original_path, image_name)
            extension = image_name.split('.')[-1]
            image_clear_name = ''.join(image_name.split('.')[:-1])
            save_name = '{}.{}'.format(image_clear_name, extension)
            save_path = os.path.join(prepared_path, save_name)
            if self.REMOVE_OLD_PREPARED or os.path.exists(save_path) is False:
                log.trash(self.LOGLEVEL, 'Start with {} of {}'.format(image_name, label))
                result_image = self._prepare_image(image_path)

                cv2.imwrite(save_path, result_image)
                log.debug(self.LOGLEVEL, '  save_path: {}'.format(save_path))

    def _prepare_labels(self):
        log.info(self.LOGLEVEL, '=======\nPrepare labels\n=======')
        for label in self.GIVEN_LABELS:
            log.debug(self.LOGLEVEL, 'Start with {}'.format(label))
            self._prepare_label(label)

    def prepare(self, remove_old_prepared=False):
        """Performs preparation.

        Args:
            remove_old_prepared (bool):

        """
        self.REMOVE_OLD_PREPARED = remove_old_prepared
        self._prepare_labels()


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
