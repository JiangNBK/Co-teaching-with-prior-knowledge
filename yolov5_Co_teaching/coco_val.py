from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab,json

if __name__ == "__main__":
    anno = COCO('E:/Noise_dataset_under_camera/data/data_set_coco/annotations/val.json')        #标注文件的路径及文件名，json文件形式
    pred = anno.loadRes('E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/runs/val/exp11/best.json')  #自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(anno, pred, "bbox")

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()