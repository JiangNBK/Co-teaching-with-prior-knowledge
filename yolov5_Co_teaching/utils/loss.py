# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""
import os
import random
import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from utils.knn import KnnRE
from utils.general import xywhn2xyxy

#还原resize前的标注比例
#label = [cx,cy,w,h]
def Re_label(label,shape,new_shape,stride=32):
    shape = np.array(shape)
    new_shape = np.array(new_shape)
    #计算收缩比
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    #计算收缩后图片的长宽
    new_unpad =   [int(round(shape[0] * r)),int(round(shape[1] * r))]
    new_wh = [int(round(shape[1] * r)),int(round(shape[0] * r))]  #新图片wh，没有unpad的
    #计算需要填充的像素
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]
    # stride表示的即是模型下采样次数的2的次方，这个涉及感受野的问题，在YOLOV5中下采样次数为5
    # 则stride为32
    #dw, dh = np.mod(dw, stride), np.mod(dh, stride) #验证时使用，训练都是resize成(640,640)
    dw /= 2  # 除以2即最终每边填充的像素
    dh /= 2
    dwh = np.array([dw,dh])
    cxy = ((label[:,:2] * new_shape[[1,0]]) - dwh) / new_wh
    wh =  (label[:,2:] * new_shape[[1,0]]) / new_wh
    return np.concatenate((cxy,wh),axis=1)




def draw_label_type(draw_img,bbox,label_color):
    label = str(bbox[-1])
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] + labelSize[0] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )
    else:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] - labelSize[1] - 3),
                      (bbox[0] + labelSize[0], bbox[1] - 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )

def show_label(path,objects):
    img = cv2.imread(path)
    H = img.shape[0]
    W = img.shape[1]
    for i, object in enumerate(objects):
        cx, cy, w, h = object[-4] * W, object[-3] * H, object[-2] * W, object[-1] * H
        box = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2),"pre"]
        box_color = (255, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=box_color, thickness=2)
    cv2.imshow("img", img)
    cv2.waitKey(0)  # 等待按键

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

#随着训练代数，knn判断为背景的逐渐为改为物体
def R_object(ob_pre,epoch,KNN=True):
    ob = ob_pre != -1
    if KNN:
        for i in range(len(ob)):
            if random.random() < 0.2+ 0.05 * epoch and ob[i] == False: ob[i] = True
        return ob
    else:
        for i in range(len(ob)):
            ob[i] = True
        return ob

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False,train_path="E:/Noise_dataset_under_camera/data/dataset_yolo/labels/train",nc= 7, iplist= ["11"]):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.KNN = KnnRE(path= train_path, nc= nc, iplist= iplist)
        self.iplist = iplist
    def __call__(self, p, targets,p_t="", epoch=0, rg_KNN=False,imgs_p="",KNN=True):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions [x,y,w,h,object,nc]
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Object regression
                if rg_KNN:
                    p_ti = p_t[i].clone().detach()
                    #th_conf=torch.tensor(epoch - 4).sigmoid() * 0.1
                    th_conf = torch.tensor(0.1)
                    ri = torch.nonzero((th_conf.item()<= p_ti[..., 4].sigmoid()) & (p_ti[..., 4].sigmoid() <=0.5), as_tuple=False)   # 阈值后面改成动态变量，得到满足阈值范围的索引
                    #ri = torch.nonzero((th_conf.item() <= p_ti[..., 4].sigmoid()),as_tuple=False)
                    imi = ri[:,0]  #图片index
                    ri_n = ri.shape[0]
                    if ri_n:
                        pre_u = p_ti[ri.T.tolist()].detach()   #不确定物体的预测
                        wh_map = torch.tensor(p_ti.shape[2:4]).cuda()
                        yx_c = ri[:, 2:]   # 预测的xy中心
                        xy_b = pre_u[:,0:2].sigmoid() * 2 - 0.5
                        xy_c = yx_c[:,[1,0]]
                        xy_u = (xy_c + xy_b) / wh_map
                        anchors_i = ri[:, 1].reshape((1,-1))
                        anchors_i = torch.cat(((torch.ones(anchors_i.shape) * i).cuda(), anchors_i), 0)
                        anchors_u = self.anchors[anchors_i.tolist()]
                        wh_u = ((pre_u[:,2:4].sigmoid() * 2) ** 2 ) * anchors_u  #不确定的wh
                        wh_r = wh_u/wh_map
                        labels = torch.cat((xy_u, wh_r), 1).cpu().numpy()
                        objects = Re_label(labels, shape=[720, 1280], new_shape=[640, 640], stride=32)
                        Im_i = torch.unique(imi)
                        nms_i = []
                        for im_i in Im_i:
                            batch_i = torch.where(imi == im_i)[0].tolist()
                            boxs = xywhn2xyxy(objects[batch_i,:])
                            nmsi = torchvision.ops.nms(torch.from_numpy(boxs).cuda(), pre_u[batch_i,:][:,4].sigmoid(), iou_threshold=0.5)
                            nms_i = nms_i + [batch_i[i] for i in nmsi.tolist()]

                        ci = [self.iplist.index(imgs_p[i].split(os.path.sep)[-1].split("_")[0]) for i in imi]
                        coh = np.identity(len(self.iplist))[ci]  #摄像头onehot
                        objects = np.concatenate((coh ,objects),axis=1) #这里多摄像头要把zero改成onehot
                        ob_pre = self.KNN.pre(objects[nms_i,:])   #判断类别
                        # show_label(path=img_p, objects=objects)
                        addobject =  ri[nms_i,:][R_object(ob_pre,epoch,KNN=KNN)]
                        #addobject = ri[nms_i, :]
                        tobj[addobject[:,0],addobject[:,1],addobject[:,2],addobject[:,3]] = 2


                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)  #计算object loss
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,cx,cy,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # ai is matrix: [[0,0,...,0], [1,1,...,1], [2,2,...,2]], ai.shape = (na, nt)
        # same as .repeat_interleave(nt),  生成shape=[na,nt],第一行全是0，第二行全是1，依此类推

        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # after repeat, targets.shape = (na, nt, 6), after cat, targets.shape = (na, nt, 7)
        # 将targets扩充至(na, nt, 7)，也就是每个anchor与每个targets都有对应，为了接下来计算损失用
        # targets的值[ [[image,class,cx,cy,w,h,0],
        #             [image,class,cx,cy,w,h,0],
        #               	...		共nt个   ]

        # 			  [[image,class,cx,cy,w,h,1]，
        #              [image,class,cx,cy,w,h,1],
        #                   ...		共nt个    ]

        # 			  [[image,class,cx,cy,w,h,2]，
        #              [image,class,cx,cy,w,h,2],
        #                   ...		共nt个    ]
        #          ]

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        # 每一层layer(共三层）单独计算。
        # 先通过判定每个target和3个anchor的长宽比是否满足一定条件，来得到满足条件的anchor所对应的targets (t)。
        # 这时的anchor数量是3，并不是某个位置的anchor，而是当前层的anchor。
        # 这时的t是3个anchor对应的targets的值，也就是说如果一个target如果对应多个anchor,那么t就有重复的值。

        # 然后根据t的每个target的中心点的偏移情况，得到扩充3倍的t。
        # 这时的t就是3个anchor对应的targets的值的扩充。

        # 接下来indices保存每层targets对应的图片索引，对应的anchor索引（只有3个），以及中心点坐标。
        # 接下来计算损失的时候，要根据targets对应的anchor索引来选择在某个具体位置的anchors,用来回归。
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare  shape = [3,n]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter  shape = [number of True, 7]
                # Offsets
                # 获取选择完成的box的中心点左边-gxy（以图像左上角为坐标原点），并转换为以特征图右下角为坐标原点的坐标-gxi
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse

                #分别判断box的（x，y）坐标是否大于1，并距离网格左上角的距离（准确的说是y距离网格上边或x距离网格左边的距离）距离小于0.5，如果（x，y）中满足上述两个条件，则选中
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  #j:x满足条件  k:y满足条件
                #对转换之后的box的（x，y）坐标分别进行判断是否大于1，并距离网格右下角的距离（准确的说是y距离网格下边或x距离网格右边的距离）距离小于0.5，如果（x，y）中满足上述两个条件，为Ture
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) #shape = [5,number of True]
                t = t.repeat((5, 1, 1))[j]  # t.repeat((5, 1, 1)).shape = [5,number of True,7]    t.shape = [,7]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]   #得到所有的锚框的偏执

            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), (grid xy), (grid wh), (anchors)
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class  分别对应所属anchor，第几张图片，类别
            gij = (gxy - offsets).long()  #得到匹配到锚框的x_index和y_index
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
