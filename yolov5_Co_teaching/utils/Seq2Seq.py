import torch
import os
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import glob
import shutil
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

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
            Labels[t_i[0]] = 0
        return Labels

class Animator:
    """For plotting data in animation."""
    def __init__(self,
                 xlabel=None, ylabel=None, legend=None, xlim=None,ylim=None,
                 xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1, ncols=1,  #行列数
                 figsize=(8, 6) #图片大小
                 ):

        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines

        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg') #Use the svg format to display a plot in Jupyter.
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self,axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.
        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):  #判断输入进来的是否为数组
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()

        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            #plt.pause(0.2)  # 暂停一秒
            #plt.ioff()
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        plt.pause(0.2)  # 暂停一秒
        plt.ioff()



class LoadCameraLabel():
    #path,class of number, list of camera_ip
    def __init__(self, p, nc, iplist = ["8","11","13"],mode="train"):
        cn = len(iplist)
        p = Path(p)
        self.f = []
        self.iplist = iplist
        self.objects = []   #摄像头ip的one-hot,x,y,h,w
        self.labels = []  #目标类别的one-hot
        self.files = []
        self.mode = mode
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
        if self.mode == "train":
            return self.objects.shape[0]
        elif self.mode == "val":
            return len(self.f)
    # obejects = []
    def __getitem__(self, index):
        if self.mode == "train":
            objects = self.objects[index]
            return objects
        elif self.mode == "val":
            filename = self.f[index]
            objects = self.objects[np.where(self.files == index)]
            labels = self.labels[np.where(self.files == index)]
            return objects, labels, Path(filename)



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax()
        )
    def forward(self, x):
        output = self.model(x)
        return output


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_path = r"E:\Noise_dataset_under_camera\data\dataset_yolo\labels\train"
    generator = Generator().to(device=device)
    lr = 0.00001
    epochs = 100
    save_p = 20
    batch_size = 4
    loss_function = nn.MSELoss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    #generator.load_state_dict(torch.load("generator.pt"))

    Dataset = LoadCameraLabel(p=train_path, nc=7, iplist = ["11"])
    train_loader = torch.utils.data.DataLoader(Dataset, batch_size=batch_size, shuffle=True)
    loss_s = []
    plt.ion()
    animator = Animator(xlabel='epoch', xlim=[0, epochs],legend=['train loss'])

    for epoch in range(epochs):
        Loss = 0
        for n, (objects) in enumerate(train_loader):
            f = objects[:,-4:]
            f = f.to(torch.float32).to(device=device)
            target = f.detach().clone()
            generator.zero_grad()
            output = generator(f)
            loss_dis = loss_function(output, target)
            loss_dis.backward()
            optimizer_g.step()
            Loss += loss_dis.item()
        loss_s.append(Loss/len(train_loader))
        if (epoch+1) % save_p == 0:
            torch.save(generator.state_dict(), f"generator_{epoch}.pt")

        animator.add(epoch,Loss/len(train_loader))
        print(f'loss {loss_dis:.3f}')
    plt.show()

class seq2seq():
    def __init__(self,weight="",train_label="E:/Noise_dataset_under_camera/data/dataset_yolo/labels/train"):
        Loss = 0
        self.model = Generator()
        self.model.load_state_dict(torch.load(weight))
        self.loss_function = nn.MSELoss()
        Dataset = LoadCameraLabel(p=train_label, nc=7, iplist=["11"],mode="train")
        train_loader = torch.utils.data.DataLoader(Dataset, batch_size=4, shuffle=True)
        for n, (objects) in enumerate(train_loader):
            f = objects[:,-4:]
            f = f.to(torch.float32)
            target = f.detach().clone()
            output = self.model(f)
            loss_dis = self.loss_function(output, target)
            Loss += loss_dis.item()
        self.loss_thu = Loss/len(train_loader)

    def pre(self,inputs):
        labels = np.ones(inputs.shape[0]) * -1
        for i in range(inputs.shape[0]):
            pre = self.model(inputs[i])
            target = inputs[i].detach().clone()
            loss = self.loss_function(pre,target)
            if loss > 1.4 * self.loss_thu:
                pass
            else:
                labels[i] = 0
        return labels

if __name__ == "__main__":
    Train = True
    if Train:
        train()
    else:
        train_path = "E:/Noise_dataset_under_camera/data/dataset_yolo/labels/train/"
        val_path = "E:/Noise_dataset_under_camera/data/dataset_yolo/labels/val/"
        val_pre_path = r"E:\Noise_dataset_under_camera\data\dataset_yolo\labels\val_pre"

        Class_dic = {-1:"background",0:"person",1:"cart",2:"forklift",3:"tractor",4:"shovel loader",5:"Furnace bottom car",6:"truck"}
        N = 0
        P = 0
        nc = 7
        iplist = ["11"]
        Method = "Seq2Seq"

        if Method == "KNN":
            #model = KnnRE(train_path, nc, iplist, val_path) #构建模型
            pass
        elif Method == "Seq2Seq":
            model = seq2seq(weight="E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/utils/generator_99.pt",train_label="E:/Noise_dataset_under_camera/data/dataset_yolo/labels/train")
        # # TSNE降维可视化
        # train_data_f = KNN.objects[:,1:]
        # train_data_l = KNN.labels
        #
        # tsne = TSNE(n_components=2, init='pca', random_state=0)
        # vis_traindata = tsne.fit_transform(train_data_f)
        # fig = plot_embedding(vis_traindata, train_data_l,"t-SNE embedding of the digits")
        # plt.show()

        testdata = LoadCameraLabel(p=val_pre_path, nc=nc, iplist=iplist,mode="val")

        for i in range(len(testdata)):
            objects, labels, file_path = testdata[i]  # 取出一个图片数据,[cx,cy,w,h],类别的one-hot
            objects = torch.from_numpy(objects[:,-4:]).to(torch.float32)
            #预测
            Pre = model.pre(objects)

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
            #Labels = np.where(Labels >= 0, Labels, 0)
            img = cv2.imread(str(image_path))
            H = img.shape[0]
            W = img.shape[1]

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
    print("Seq2Seq:",P/N)