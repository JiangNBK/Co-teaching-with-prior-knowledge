import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


#添加中文
def cv2AddChineseText(img, text, position, textColor=(255, 0, 0), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

#计算距离
def dis(pts):
    h1 = np.linalg.norm(pts[0]-pts[1])
    h2 = np.linalg.norm(pts[2] - pts[3])
    w1 = np.linalg.norm(pts[1] - pts[2])
    w2 = np.linalg.norm(pts[3] - pts[0])
    return int((w1+w2)/2),int((h1+h2)/2)

#坐标转换
def cvt_pos(u , v, mat):
    x = (mat[0][0]*u+mat[0][1]*v+mat[0][2])/(mat[2][0]*u+mat[2][1]*v+mat[2][2])
    y = (mat[1][0]*u+mat[1][1]*v+mat[1][2])/(mat[2][0]*u+mat[2][1]*v+mat[2][2])
    return x, y

#molten iron 铁水增强
def MI(src,labels):
    if labels.ndim == 1: labels = np.array([labels])
    H, W = src.shape[:-1]
    src_fire = np.copy(src)
    fire = cv2.imread("./data/fire_img/epoch500.jpg")
    for label in labels:
        if random.random() <= 0.2:
            w, h = int(label[3] * W), int(label[4] * H)
            x, y = int(label[1] * W), int(label[2] * H)
            re_fire = cv2.resize(fire, (w, h), interpolation=cv2.INTER_AREA)
            src_fire[int(y - h / 2):int(y - h / 2)+h, int(x - w / 2):int(x - w / 2)+w] = re_fire
            src = cv2.addWeighted(src, 0.7, src_fire, 0.3, 0)
    return src


#将火焰添加到背景上
def fire_aug(src,labels,img_n,w = "random"):
    result = np.copy(src)
    #if random.random()<=0.5:
    if True:
        if w == "random":
            n = random.randint(1, 10)
            fire = cv2.imread("./data/fire_img/526_epoch200.jpg")
            #fire = np.float32(fire)/255
            H,W = src.shape[0], src.shape[1]
            fire = cv2.resize(fire, (W,H))
            fire_img = cv2.addWeighted(src, random.uniform(0.5, 1), fire, random.uniform(0.4, 0.6), 0)
            xywhs = np.random.random((n,4))
            xywhs[:, [0, 2]] = xywhs[:, [0, 2]] * W
            xywhs[:, [1, 3]] = xywhs[:, [1, 3]] * H
            xyxys = xywh2xyxy(xywhs).astype(np.int16)
            xyxys[:, [0, 2]] = np.clip(xyxys[:, [0, 2]], 0, W)
            xyxys[:, [1, 3]] = np.clip(xyxys[:, [1, 3]], 0, H)
            for i in range(n):
                x1,y1,x2,y2 = [k for k in xyxys[i]]
                result[y1:y2, x1:x2] = fire_img[y1:y2, x1:x2]
            return result
        if w == "small":
            n_patch = 16
            p = 0.4
            n = int(n_patch **2 * p)
            indexs = np.linspace(0,n_patch **2-1,n_patch **2,dtype=int)
            index = np.random.choice(indexs, size=n, replace=False)
            fire = cv2.imread("./data/fire_img/526_epoch200.jpg")
            H, W = src.shape[0], src.shape[1]
            fire = cv2.resize(fire, (W, H))
            fire_img = cv2.addWeighted(src, random.uniform(0.5, 1), fire, random.uniform(0.2, 0.6), 0)

            xs = np.linspace(0, W, n_patch + 1, dtype=np.int16)
            ys = np.linspace(0, H, n_patch + 1, dtype=np.int16)

            for i in range(n_patch):
                x1 = xs[:-1]
                y1 = np.ones(n_patch)* ys[i]
                x2 = xs[1:]
                y2 = np.ones(n_patch) * ys[i+1]
                if i == 0:
                    xyxys = np.concatenate(([x1], [y1], [x2], [y2]), axis=0).T
                else:
                    xyxy = np.concatenate(([x1], [y1], [x2], [y2]), axis=0).T
                    xyxys = np.concatenate((xyxys,xyxy), axis=0)

            xyxys = xyxys[index].astype(np.int16)
            xyxys[:, [0, 2]] = np.clip(xyxys[:, [0, 2]], 0, W)
            xyxys[:, [1, 3]] = np.clip(xyxys[:, [1, 3]], 0, H)
            for i in range(n):
                x1, y1, x2, y2 = [k for k in xyxys[i]]
                result[y1:y2, x1:x2] = fire_img[y1:y2, x1:x2]
            return result
        if w == "large":
            n_patch = 8
            p = 0.4
            n = int(n_patch **2 * p)
            indexs = np.linspace(0,n_patch **2-1,n_patch **2,dtype=int)
            index = np.random.choice(indexs, size=n, replace=False)
            fire = cv2.imread("./data/fire_img/526_epoch200.jpg")
            H, W = src.shape[0], src.shape[1]
            fire = cv2.resize(fire, (W, H))
            fire_img = cv2.addWeighted(src, random.uniform(0.5, 1), fire, random.uniform(0.2, 0.6), 0)

            xs = np.linspace(0, W, n_patch + 1, dtype=np.int16)
            ys = np.linspace(0, H, n_patch + 1, dtype=np.int16)

            for i in range(n_patch):
                x1 = xs[:-1]
                y1 = np.ones(n_patch)* ys[i]
                x2 = xs[1:]
                y2 = np.ones(n_patch) * ys[i+1]
                if i == 0:
                    xyxys = np.concatenate(([x1], [y1], [x2], [y2]), axis=0).T
                else:
                    xyxy = np.concatenate(([x1], [y1], [x2], [y2]), axis=0).T
                    xyxys = np.concatenate((xyxys,xyxy), axis=0)

            xyxys = xyxys[index].astype(np.int16)
            xyxys[:, [0, 2]] = np.clip(xyxys[:, [0, 2]], 0, W)
            xyxys[:, [1, 3]] = np.clip(xyxys[:, [1, 3]], 0, H)
            for i in range(n):
                x1, y1, x2, y2 = [k for k in xyxys[i]]
                result[y1:y2, x1:x2] = fire_img[y1:y2, x1:x2]
            return result

        if w == "medium":
            n_patch = 12
            p = 0.4
            n = int(n_patch **2 * p)
            indexs = np.linspace(0,n_patch **2-1,n_patch **2,dtype=int)
            index = np.random.choice(indexs, size=n, replace=False)
            fire = cv2.imread("../data/fire_img/526_epoch200.jpg")
            H, W = src.shape[0], src.shape[1]
            fire = cv2.resize(fire, (W, H))
            fire_img = cv2.addWeighted(src, random.uniform(0.5, 1), fire, random.uniform(0.2, 0.6), 0)

            xs = np.linspace(0, W, n_patch + 1, dtype=np.int16)
            ys = np.linspace(0, H, n_patch + 1, dtype=np.int16)

            for i in range(n_patch):
                x1 = xs[:-1]
                y1 = np.ones(n_patch)* ys[i]
                x2 = xs[1:]
                y2 = np.ones(n_patch) * ys[i+1]
                if i == 0:
                    xyxys = np.concatenate(([x1], [y1], [x2], [y2]), axis=0).T
                else:
                    xyxy = np.concatenate(([x1], [y1], [x2], [y2]), axis=0).T
                    xyxys = np.concatenate((xyxys,xyxy), axis=0)

            xyxys = xyxys[index].astype(np.int16)
            xyxys[:, [0, 2]] = np.clip(xyxys[:, [0, 2]], 0, W)
            xyxys[:, [1, 3]] = np.clip(xyxys[:, [1, 3]], 0, H)
            for i in range(n):
                x1, y1, x2, y2 = [k for k in xyxys[i]]
                result[y1:y2, x1:x2] = fire_img[y1:y2, x1:x2]
            return result
        if w == "Flame_region":
            fire_s = ["epoch100.jpg","epoch800.jpg","526_epoch200.jpg","526_epoch2400.jpg"]
            fire_s = random.choices(fire_s, k=2)
            for f_p in fire_s:
                fire = cv2.imread(f"E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/data/fire_img/{f_p}")
                H, W = src.shape[0], src.shape[1]
                fire = cv2.resize(fire, (W, H))
                # region_path = ["11_81_mask.jpg","11_104_mask.jpg","11_119_mask.jpg","11_187_mask.jpg","11_191_mask.jpg"]
                region_path = os.listdir("E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/data/mask_region/")
                for r_p in region_path:
                    src_ori = np.copy(src)
                    region = cv2.imread(f"E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/data/mask_region/{r_p}",cv2.IMREAD_GRAYSCALE)
                    fire_img = cv2.addWeighted(src_ori, random.uniform(0.5, 1), fire, random.uniform(0.8, 1.1), 0)
                    src_ori[region>=250] = fire_img[region>=250]
                    cv2.imwrite(f"C:/Users/jnbk/Desktop/FA_image/{img_n+f_p+r_p} .jpg", src_ori)
            return result
    else:
        return src


#safety warning projector 安全警示投光仪
def Swp(img,labels):
    img = np.float32(img)
    ih,iw  = img.shape[:-1]
    if labels.ndim == 1: labels = np.array([labels])
    x,y,w,h = np.split(labels[:,1:],(1,2,3),axis=1)
    brightness = 70
    up = cv2.convertScaleAbs(img, beta=brightness)
    # 坐标点points
    X = (x + np.random.uniform(0, 0.02, size=x.shape)) * iw
    Y = (y + np.random.uniform(0, 0.02, size=y.shape)) * ih
    W = (w + np.random.uniform(0, 0.02, size=w.shape)) * iw
    H = (h + np.random.uniform(0, 0.02, size=h.shape)) * ih
    pts = np.array([
                    X-W/2, Y-H/2,
                    X+W/2, Y-H/2,
                    X+W/2, Y+H/2,
                    X-W/2, Y+H/2
                    ]).T.reshape((-1,4,2))
    if labels.ndim == 1:pts = np.array([pts])

    pts = np.around(pts)
    # 和原始图像一样大小的0矩阵，作为mask
    mask = np.zeros(img.shape[:2], np.uint8)
    txt_L = np.zeros(img.shape).astype(np.uint8)
    for i, p in enumerate(pts):
        if random.random()<=0.2:
            h, w, c = img.shape  # h=240  w=320
            # 得到转换矩阵
            #img_t = cv2.imread("E:/Noise_dataset_under_camera/data/dataset_yolo/images/train/11_187.jpg")
            a_w,a_h = labels[i,3]*w,labels[i,4]*h
            p_w = a_w/90
            if p_w > 1: p_w = 1
            src_list = [(310, 198), (310 + 228 * p_w, 198 + 89 * p_w), (310 + 334 * p_w, 198 + 51 * p_w),(310 + 90 * p_w, 198 - 26 * p_w)]
            pts1 = np.float32(src_list)
            nw, nh = dis(pts1)
            txt_l = np.zeros((nh, nw, 3)).astype(np.uint8)
            # txt_l = cv2AddChineseText(txt_l, "行", (int(nw *2/ 6), int(nh * 1 / 10)))
            # txt_l = cv2AddChineseText(txt_l, "车", (int(nw *2/ 6), int(nh * 3 / 10)))
            # txt_l = cv2AddChineseText(txt_l, "运", (int(nw *2/ 6), int(nh * 5 / 10)))
            # txt_l = cv2AddChineseText(txt_l, "行", (int(nw *2/ 6), int(nh * 7 / 10)))

            cx = (p[0,0]+p[2,0])/2
            cy = (p[0,1]+p[2,1])/2
            pts1[:, 0] = pts1[:, 0] - ((310+310 + 334 * p_w)/2-cx)
            pts1[:, 1] = pts1[:, 1] - ((198+198 + 51 * p_w)/2-cy)

            pts1[:, 0]= np.clip(pts1[:, 0], 0, img.shape[1])
            pts1[:, 1] = np.clip(pts1[:, 1], 0, img.shape[0])

            #pts1=  np.float32(([[53,35],[281,124],[384,86],[143,9]])

            pts2 = np.float32([[0, 0], [0, nh], [nw, nh],[nw, 0]])
            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            txt_L += cv2.warpPerspective(txt_l, matrix, (w, h))

            # 在mask上将多边形区域填充为白色
            pts1 = np.array([pts1]).astype(int)
            cv2.polylines(mask, pts1, 1, 255)  # 描绘边缘

            cv2.fillPoly(mask, pts1, 255)  # 填充

            cv2.polylines(img, pts1, 0, 0)

            cv2.fillPoly(img, pts1, 0)

    # 逐位与，得到裁剪后图像，此时是黑色背景
    dst = cv2.bitwise_and(up, up, mask=mask)

    # 添加白色背景
    result = img + dst
    index = txt_L[:,:,2] >= 1
    result[index] = [128, 128, 240]
    return result


#white light
def Wl(img,x,y,w,h):
    pass
    return img



def stell_aug(src,labels,img_n):
    #up = Swp(src, labels)
    #up1 = MI(up, labels)
    up2 = fire_aug(src, labels,img_n,w="Flame_region")
    return up2

if __name__ == "__main__":
    imgs = ["11_122","11_297","11_386","11_521","11_694"]
    for img_n in imgs:
        p = f"E:/Noise_dataset_under_camera/data/dataset_yolo/images/train/{img_n}.jpg"
        l = f"E:/Noise_dataset_under_camera/data/dataset_yolo/labels/train/{img_n}.txt"
        src = cv2.imread(p)
        labels = np.loadtxt(l)
        up1 = stell_aug(src,labels,img_n)
    # cv2.imwrite("C:/Users/jnbk/Desktop/FA_image/FA_11_297.jpg", up1)
    # up1 = cv2.resize(up1, (640,640))
    # cv2.imwrite("C:/Users/jnbk1/Desktop/random.jpg", up1)
    # cv2.imshow('up', up1)
    # cv2.waitKey()


