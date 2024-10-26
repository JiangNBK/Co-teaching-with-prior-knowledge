import cv2
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """

    :param x: 坐标点
    :param img: 图片
    :param color: 三原色值
    :param label: 标注名
    :param line_thickness:  线框厚度
    :return:
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


img = cv2.imread(r"E:\Noise_dataset_under_camera\data\dataset_yolo\images\val\11_0.jpg")
x = [100,100,500,500]
color = (0,255,255)
label = "person"
line_thickness = 4


plot_one_box(x,img,color,label,line_thickness)
