import experiment, log, utils, learning


class Evaluator:
    def __init__(self, expn, exp_path, Config_class, model, loglevel=log.NO):
        self.EXPN = expn
        self.EXP_PATH = exp_path
        self.CONFIG_CLASS = Config_class
        self.LOGLEVEL = loglevel
        self.MODEL = model


    def evaluate(self, label, selected_energy_levels, test_batch_size=50):
        prefixed = label + '_'
        given_labels = list(filter(lambda x: x.startswith(prefixed), utils.listdir(self.CONFIG_CLASS().RAW_PATH)))
        evaluation = dict()

        for tmp_label in given_labels:
            if tmp_label == label + '_0':
                self.CONFIG_CLASS.SPLITTING_INFO = {
                    'regular': [tmp_label]
                }
            else:
                self.CONFIG_CLASS.SPLITTING_INFO = {
                    'chaotic': [tmp_label]
                }
            val_exper = experiment.Experiment(self.EXPN, self.EXP_PATH, self.CONFIG_CLASS (), self.LOGLEVEL)
            val_exper.prepare_validation()
            val_loader = val_exper.get_validation_loader(test_batch_size, selected_energy_levels)
            evaluation[tmp_label] = learning.test(self.MODEL, val_loader)
            val_exper.remove()
        return evaluation


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
