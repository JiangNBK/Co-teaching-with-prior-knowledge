_base_ = '../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'

max_epochs = 300  # 训练的最大 epoch
# -----data related-----
data_root = '/home/user02/mmyolo/data/coco/' # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/clear_train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/val.json'
val_data_prefix = 'images/val/'  # Prefix of val image path

img_scale = [640,640]

class_name = ('person', 'car', 'ft', 'tlj', 'sl', 'ldc', 'tk', )  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=1,  # 第几个 epoch 后验证，这里设置 20 是因为前 20 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=10  # 每 val_interval 轮迭代进行一次测试评估
)


pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]


mosaic_transform= [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.1, 1.9),  # scale = 0.9
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform,
    *mosaic_transform,
    dict(
        type='YOLOv5MixUp',
        prob=0.1,
        pre_transform=[
            *pre_transform,
            *mosaic_transform
        ]),
    ...
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        metainfo=metainfo,
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        pipline = train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32)
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=metainfo,
        type='YOLOv5CocoDataset',
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file
        ))

test_dataloader = val_dataloader



default_hooks = dict(
    # interval设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        max_keep_ckpts=5,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    # logger 输出的间隔
    logger=dict(type='LoggerHook', interval=10))