import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sn
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import contextlib
import os
from matplotlib.font_manager import FontProperties
plt.rc('font', family='Times New Roman')
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 16
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y



def plot_labels(labels, names=(), save_dir=Path(''),classname="object",color="blue"):
    # plot dataset labels
    colors = Colors()
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['X', 'Y', 'Width', 'Height'])

    # seaborn correlogram
    # #相关性分析
    # sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    # plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    # plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)[1].ravel()
    # y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # with contextlib.suppress(Exception):  # color histogram bars by class
    #     [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    # ax[0].set_ylabel('instances')
    # if 0 < len(names) < 30:
    #     ax[0].set_xticks(range(len(names)))
    #     ax[0].set_xticklabels(list(names), rotation=90, fontsize=10)
    # else:
    #     ax[0].set_xlabel('classes')
    sn.histplot(x, x='X', y='Y', ax=ax[0], bins=50, pmax=0.9, color=color)
    sn.histplot(x, x='Width', y='Height', ax=ax[1], bins=50, pmax=0.9, color=color)

    font = FontProperties(weight='bold')
    # 设置 x 轴和 y 轴标签，应用加粗字体
    ax[0].set_xlabel('X',fontproperties=font)
    ax[0].set_ylabel('Y',fontproperties=font)

    ax[1].set_xlabel('Width', fontproperties=font)
    ax[1].set_ylabel('Height', fontproperties=font)

    # 设置x轴和y轴标签只显示一位小数
    ax[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax[1].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # # 设置坐标轴的等比例显示
    # plt.axis('equal')
    # # 设置坐标轴的等比例显示
    # plt.gca().set_aspect('equal')
    #ax[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)

    # 前1000个框可视化
    # for cls, *box in labels[:1000]:
    #     ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    # ax[1].imshow(img)
    # ax[1].axis('off')

    #隐藏边框
    # for a in [0, 1]:
    #     for s in ['top', 'right', 'left', 'bottom']:
    #         ax[a].spines[s].set_visible(False)
    #plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.savefig(save_dir / f'{classname}labels.jpg', dpi=200, bbox_inches='tight', pad_inches=0.01)
    matplotlib.use('Agg')
    plt.close()


def parse_yolo_txt_file(txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        line = line.strip().split()
        class_label = int(line[0])
        x_center = float(line[1])
        y_center = float(line[2])
        width = float(line[3])
        height = float(line[4])

        annotation = [class_label, x_center, y_center, width, height]
        annotations.append(annotation)

    return np.array(annotations)

if __name__ == "__main__":
    folder_path = 'E:/Noise_dataset_under_camera/data/dataset_yolo/labels/train'  # 指定包含 YOLO 格式文本文件的文件夹路径
    file_extension = '.txt'  # 文件扩展名

    txt_files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]

    all_annotations = []
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        annotations = parse_yolo_txt_file(file_path)
        all_annotations.append(annotations)

    all_annotations = np.concatenate(all_annotations, axis=0)

    class_dict = {0:"Person",2:"Forklift",3:"Tractor",4:"Shovel_loader",5:"Furnace_bottom_car"}
    colors = ['#FF5D5D', '#F4800C', '#F5B239', '#F5DC27', '#98F056']
    k = 0
    for i in class_dict.keys():
        filtered_array = all_annotations[all_annotations[:, 0] == i]
        plot_labels(labels=filtered_array, names=(0,1,2,3,4,5,6), save_dir=Path('C:/Users/jnbk1/Desktop/论文/小论文/基于位置先验信息的钢铁厂半监督目标检测/Dataset'),classname=class_dict[i],color=colors[k])
        k += 1
