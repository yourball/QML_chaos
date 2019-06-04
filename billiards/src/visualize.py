import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    npimg = img.numpy().astype('float32')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_random_images(train_loader):
    dataiter = iter(train_loader)
    print (dataiter)
    images, labels, energy_levels = dataiter.next()
    
    labels = labels.type(torch.uint8)
    inv_labels = (labels - 1) / 255

    reg_images = images[labels]
    ch_images = images[inv_labels]

    if len(reg_images) > 0:
        plt.title('Regular')
        imshow(torchvision.utils.make_grid(reg_images))
    if len(ch_images) > 0:
        plt.title('Chaotic')
        imshow(torchvision.utils.make_grid(ch_images))

    return reg_images, ch_images


def plot_predicted_images(model, test_loader):
    dataiter = iter(test_loader)
    images_test, labels_test, _ = dataiter.next()
    #images_test = images_test.long ()
    labels_test = labels_test.long ()
	
    inputs = Variable(images_test)
    outputs = model(inputs)

    pred = outputs.max(1, keepdim=True)[1]

    corr_predict = labels_test == pred.view(-1)
    wr_predict = labels_test != pred.view(-1)

    if len(images_test[corr_predict]) > 0:
        plt.title('Correct')
        imshow(torchvision.utils.make_grid(images_test[corr_predict]))
    if len(images_test[wr_predict]) > 0:
        plt.title('Wrong')
        imshow(torchvision.utils.make_grid(images_test[wr_predict]))


def plot_weights(model):
    for filter_weigths in list(model.parameters())[0].detach().numpy():
        filter_weigths = filter_weigths[0]
        plt.imshow(filter_weigths)
        plt.colorbar()
        plt.show()

    for filter_weigths in list(model.parameters())[2].detach().numpy():
        filter_weigths = filter_weigths[0]
        plt.imshow(filter_weigths)
        plt.colorbar()
        plt.show()


def plot_evaluation(eval_stats, lmbda_lim = 0.4, xlabel=None, filename=None):
    labels_list = list(eval_stats.keys())
    label_list = labels_list.sort()
    params = list(map(lambda x: float(x.split('_')[-1]), labels_list))
    
    params_mapping = dict(zip(labels_list, params))

    activations_one = list()
    activations_zero = list()
    params_plot = list()
    
    ll = list(eval_stats.keys())
    ll = np.array(ll)
    for label in np.sort(ll):
        activations_zero.append(eval_stats[label][2])
        activations_one.append(eval_stats[label][3])
        params_plot.append(params_mapping[label])

    if xlabel:
        plt.xlabel(xlabel)

    plt.plot(params_plot, activations_zero, label=r'$chaotic$', linestyle='-', marker='o', markersize = 8)
    plt.plot(params_plot, activations_one, label=r'$regular$', linestyle='-', marker='o', markersize = 8)
    
    plt.xlabel(r'$\lambda$', fontsize=16)
    plt.ylabel('NN output', fontsize=16)    
    plt.xlim([0,lmbda_lim])
    plt.legend(fontsize=16)

    if filename:
        plt.savefig(filename + '.eps', format='eps', dpi=1000)
        plt.savefig(filename + '.pdf', format='pdf', dpi=1000)
    
    return params_plot, activations_one, activations_zero 

def plot_test(test_losses, test_accuracies):
    test_accuracies_toplot = np.array(test_accuracies) / 100.
    plt.plot(test_accuracies_toplot, label='Accuracy', color='orange')
    plt.plot(test_losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
