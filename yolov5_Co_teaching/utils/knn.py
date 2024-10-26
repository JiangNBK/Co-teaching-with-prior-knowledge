import cv2
import glob
import shutil
import os
import torch
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
from sklearn import neighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


#计算IoU值


def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)

    bbox1 = xywh2xyxy(bbox1)
    bbox2 = xywh2xyxy(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union



def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    print(b1_x2)
    print(b2_x2)
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)


    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf

    return iou.numpy()  # IoU





def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    :param x: 坐标点 例：x = [100,100,500,500]
    :param img: 图片
    :param color: 三原色值
    :param label: 标注名
    :param line_thickness:  线框厚度
    :return:
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img





class LoadCameraLabel():
    #path,class of number, list of camera_ip
    def __init__(self, p, nc, iplist = ["8","11","13"]):
        cn = len(iplist)
        p = Path(p)
        self.f = []
        self.iplist = iplist
        self.objects = []   #摄像头ip的one-hot,x,y,h,w
        self.labels = []  #目标类别的one-hot
        self.files = []
        if p.is_dir():  # dir
            self.f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        pbar = enumerate(self.f)
        pbar = tqdm(pbar, total=len(self.f), ncols=shutil.get_terminal_size().columns,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, file in pbar:
            name = file.split(os.path.sep)[-1]
            ip = name.split("_")[0]
            one_hot_index = self.iplist.index(ip)  #当前ip的one-hot index
            l = np.loadtxt(file).reshape(-1,5) #标注
            c = np.zeros((l.shape[0],cn))
            c [:,one_hot_index ]=1  #摄像头独热编码
            file_index = np.ones(l.shape[0]) * i
            if i == 0:
                self.objects= np.concatenate((c,l[:,1:]),axis=1)
                self.labels = np.identity(nc)[l[:,0].astype(int)]
                self.files = file_index
            else:
                newobject = np.concatenate((c, l[:, 1:]), axis=1)
                self.objects = np.concatenate((self.objects, newobject), axis=0)
                newlabel = np.identity(nc)[l[:,0].astype(int)]
                self.labels = np.concatenate((self.labels, newlabel), axis=0)
                self.files = np.concatenate((self.files,file_index))

        self.labels = np.concatenate((self.labels, np.ones((self.labels.shape[0],1))), axis=1)
    def __len__(self):
        return len(self.f)

    # obejects = []
    def __getitem__(self, index):
        filename = self.f[index]
        objects = self.objects[np.where(self.files==index)]
        labels = self.labels[np.where(self.files==index)]

        return objects,labels,Path(filename)


class KnnRE():
    def __init__(self,path, nc, iplist, valpath=""):
        dataset = LoadCameraLabel(p=path, nc=nc, iplist=iplist) #构建数据集class

        self.objects = dataset.objects #摄像头ip的one-hot,x,y,h,w
        self.labels_onehot = dataset.labels #目标类别的one-hot
        self.labels =np.argmax(self.labels_onehot, axis=1) #目标类别的number
        self.k = 7
        self.α = 0.8
        # 建立模型
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=self.k ,
                                             weights='uniform',
                                             algorithm='auto',
                                             leaf_size=30,
                                             p=2,
                                             metric='minkowski',
                                             metric_params=None,
                                             n_jobs=1)
        # 训练模型
        trainX = self.objects
        trainY = np.array(self.labels).reshape(-1)
        self.clf.fit(trainX, trainY)
        print("训练准确率:" + str(self.clf.score(trainX,trainY)))

        if valpath:
            val_dataset = LoadCameraLabel(p=valpath, nc=nc, iplist=iplist)
            self.val_objects = val_dataset.objects  # 摄像头ip的one-hot,x,y,h,w
            self.val_labels_onehot = val_dataset.labels  # 目标类别的one-hot
            self.val_labels = np.argmax(self.val_labels_onehot, axis=1)  # 目标类别的number
            val_trainX = self.val_objects
            val_trainY = np.array(self.val_labels).reshape(-1)
            print("验证集准确率:" + str(self.clf.score(val_trainX, val_trainY)))
    #输入为[[摄像头ip的one-hot,x,y,h,w]] shape = [n,(number of camera_ip + 4)]
    #输出为shape = [n]
    def pre(self,objects):
        # 预测的类别
        pre = self.clf.predict(objects)
        # 预测点的最近k个点，并返回距离，注：objects包含有多个物体，所以返回的pre_dis和indexs也有多个
        pre_dis, indexs = self.clf.kneighbors(X=objects, n_neighbors=self.k , return_distance=True)
        #计算最近每个物体的k个点的平均距离
        dis_means = np.mean(pre_dis, axis=1)
        #计算距离阈值
        dis_thres = self.Dynamic_threshold(indexs)
        Pre = np.ones((len(objects))) * -1
        Pre[self.α * dis_means <= dis_thres] = pre[self.α * dis_means <= dis_thres]
        return Pre

    def Dynamic_threshold(self, indexs):
        ths = []
        for ins in indexs:
            dis = self.clf.kneighbors(X=self.objects[ins], n_neighbors=self.k , return_distance=True)[0]
            th = dis.mean()
            ths.append(th)
        return np.array(ths).reshape(-1)



def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def B_labels(pre_f,val_labels):
    if val_labels.size == 0:
        Labels = np.ones(pre_f.shape[0]) * -1.0
        return Labels
    IOUS = calc_iou(pre_f, val_labels[:, 1:])
    T_index = []
    Labels = np.ones(pre_f.shape[0]) * -1.0
    if IOUS.size == 0:
        return Labels
    else:
        while True:
            m = np.max(IOUS)
            if m <=0.45: break
            index = np.where(IOUS==m)
            T_index.append(index)
            IOUS[index[0], :] = 0
            IOUS[:, index[1]] = 0
        for t_i in T_index:
            Labels[t_i[0]] = val_labels[t_i[1],0]
        return Labels



if __name__ == "__main__":
    train_path = r"E:\Noise_dataset_under_camera\data\dataset_yolo\labels\train"
    val_path = "E:/Noise_dataset_under_camera/data/dataset_yolo/labels/val/"
    val_pre_path = r"E:\Noise_dataset_under_camera\data\dataset_yolo\labels\val_pre"

    Class_dic = {-1:"background",0:"person",1:"cart",2:"forklift",3:"tractor",4:"shovel loader",5:"Furnace bottom car",6:"truck"}
    N = 0
    P = 0
    nc = 7
    iplist = ["11"]
    Method = "KNN"

    if Method == "KNN":
        KNN = KnnRE(train_path, nc, iplist, val_path) #构建模型
    elif Method == "Seq2Seq":
        pass
    # # TSNE降维可视化
    # train_data_f = KNN.objects[:,1:]
    # train_data_l = KNN.labels
    #
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # vis_traindata = tsne.fit_transform(train_data_f)
    # fig = plot_embedding(vis_traindata, train_data_l,"t-SNE embedding of the digits")
    # plt.show()

    testdata = LoadCameraLabel(p=val_pre_path, nc=nc, iplist=iplist)

    for i in range(len(testdata)-1):
        objects, labels, file_path = testdata[i]  # 取出一个图片数据,[cx,cy,w,h],类别的one-hot

        #预测
        if Method == "KNN":
            Pre = KNN.pre(objects)

        file_path = list(file_path.parts)
        label_name = file_path[-1]
        label_path = val_path + label_name
        file_path[-3] = "images"
        file_path[-2] = "val"
        image_path = Path(*file_path).with_suffix(".jpg")

        if os.path.exists(label_path):
            Label = np.loadtxt(label_path)  #读取验证集数据
            if Label.ndim == 1: Label = np.array([Label])
        else:
            Label = np.array([])

        Labels = B_labels(objects[:,-4:],Label)

        if os.path.exists(str(image_path)):
            img = cv2.imread(str(image_path))
            H = img.shape[0]
            W = img.shape[1]
        else:
            H = 720
            W = 1280

        N += Labels.shape[0]
        P += np.sum(Pre == Labels)

        # #查看验证集预测效果
        # for i, object in enumerate(objects):
        #     cx, cy, w, h = object[-4] * W, object[-3] * H, object[-2] * W, object[-1] * H
        #     box = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
        #     label = "pre:" + Class_dic[Pre[i]] + "label:" + Class_dic[Labels[i]]
        #
        #     img = plot_one_box(box,img,color=(255, 255, 0),label=label,line_thickness=2)
        # # for i, object in enumerate(Label):
        # #     cx, cy, w, h = object[-4] * W, object[-3] * H, object[-2] * W, object[-1] * H
        # #     box = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
        # #     label = Class_dic[object[0]]
        # #
        # #     img = plot_one_box(box, img, color=(255, 255, 0), label=label, line_thickness=2)

        # cv2.imshow(label_name, img)
        # cv2.waitKey(0)  # 等待按键
print("KNN准确率:",P/N)
