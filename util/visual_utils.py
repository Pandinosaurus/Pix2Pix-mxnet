import numpy as np
from matplotlib import pyplot as plt


def visualize(img_arr):
    plt.imshow(img_arr)
    plt.axis('off')
    plt.show()
    plt.pause(0.05)


def preview_train_data(train_data):
    img_in_list, img_out_list = train_data.next().data
    for i in range(4):
        plt.subplot(2, 4, i+1)
        visualize(img_in_list[i])
        plt.subplot(2, 4, i+5)
        visualize(img_out_list[i])
    plt.show()
