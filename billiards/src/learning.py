import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re

from . import log, utils, learning, experiment, visualize


def evaluate_label(label, model, test_batch_size, config, selected_energy_levels, loglevel=log.NO):
    labels_list = list(filter(lambda x: x.startswith(label + '_'), utils.listdir(config.RAW_PATH)))
    evaluation = dict()
    config.LOGLEVEL = loglevel

    for label_eval in labels_list:
        if label_eval == label + '_0':
            config.SPLITTING_INFO = {
                'regular': [label_eval]
            }
        else:
            config.SPLITTING_INFO = {
                'chaotic': [label_eval]
            }
        config.GIVEN_LABELS = [label_eval]
        config.BOUNDING_LABELS = [label_eval]
        config.update()

        exper = experiment.Experiment(config)
        exper.slice(True)
        exper.augment(True)
        exper.dump(True)

        # exper.slice()
        # exper.augment()
        # exper.dump()

        exper.validation()
        data_loader = learning.get_data_loader(exper.config.EXPN, test_batch_size,
                                               selected_energy_levels)
        evaluation[label_eval] = learning.test(model, data_loader)
        exper.remove()

    return evaluation


def evaluate(model, test_batch_size, config, selected_energy_levels, loglevel=log.NO):
    i = 1
    plt.figure(figsize=(15, 15))
    for label in ['bunimovich', 'cardioid', 'sinai', 'bunimovich2']:
        eval_stats = evaluate_label(label, model, test_batch_size, config,
                                    selected_energy_levels, loglevel)
        plt.subplot(2, 2, i)
        plt.title(label)
        visualize.plot_evaluation(eval_stats)
        i += 1
    plt.show()


def train(model, train_loader, optimizer, epoch, log_interval=20):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        target = target.long ()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if log_interval != -1 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, loglevel=log.NO):
    model.eval()
    test_loss = 0
    correct = 0

    mean_zero = 0.
    mean_one = 0.

    with torch.no_grad():
        for data, target, _ in test_loader:
            target = target.long()
            output = model(data)

            mean_zero += output[:, 0].detach().numpy().sum()
            mean_one += output[:, 1].detach().numpy().sum()

            test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).item()  # Sum up batch loss.
            pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    log.info(loglevel, 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
             test_loss, correct, len(test_loader.dataset), accuracy))

    mean_zero /= len(test_loader.dataset)
    mean_one /= len(test_loader.dataset)

    return test_loss, accuracy, mean_zero, mean_one


def exp_info(EXPN):
    with open('data/experiments/{}/info.txt'.format(str(EXPN)), 'r') as f:
        info_txt = f.read()
    expName = re.search(r'NAME.*', info_txt)
    expName = ' '.join(expName.group(0).split()[1:])
    return info_txt, expName


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
