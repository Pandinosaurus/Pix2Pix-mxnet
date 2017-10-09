import cv2


def show_mxnet_to_numpy_array(image_name, image_array):

    print(image_array.shape)

    cv2.imshow(image_name, cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


def show_mxnet_array(image_array, image_name):
    cv2.imshow(image_name, cv2.cvtColor(image_array.asnumpy(), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)