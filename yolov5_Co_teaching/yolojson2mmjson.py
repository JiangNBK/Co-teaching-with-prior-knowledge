from pycocotools.coco import COCO
import json



gt_path = 'E:/Noise_dataset_under_camera/data/data_set_coco/annotations/val.json'
result_path = 'E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/runs/val/exp11/best_predictions.json'
cocoGt = COCO(gt_path)  # 导入gt

result = open(result_path,'r',encoding='utf-8')
yolo_results = json.load(result)

result = open(result_path,'r',encoding='utf-8')
yolo_results1 = json.load(result)
print(len(yolo_results1))
imgIds = cocoGt.getImgIds()  # 图片id
k = 0
indexs=[]
for imgId in imgIds:
    rs = enumerate(yolo_results)
    img = cocoGt.loadImgs(imgId)[0]
    W = img['width']
    H = img['height']
    imgNm = img['file_name'].split('.')[0]
    for i, r in rs:
        if r["image_id"] == imgNm:
            yolo_results1[i]["image_id"] = int(imgId)
            k+=1
            indexs.append(i)

yolo_results2 = []
for i in indexs:
    yolo_results2.append(yolo_results1[i])
yolo_results2= json.dumps(yolo_results2)
f2 = open("E:/image-learning/Python/co-teaching for yolov5/yolov5_knn/runs/val/exp11/best.json", 'w')
f2.write(yolo_results2)
f2.close()