import cv2
from matplotlib import pyplot as plt


def visualize_cv2(label, img_arr):
    cv2.imshow(label, cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


def preview_train_data(train_data):
    img_in_list, img_out_list = train_data.next().data
    for i in range(4):
        plt.subplot(2, 4, i+1)
        visualize(img_in_list[i])
        plt.subplot(2, 4, i+5)
        visualize(img_out_list[i])
    plt.show()
