import cv2


def visualize_cv2(label, img_arr):
    cv2.imshow(label, cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
