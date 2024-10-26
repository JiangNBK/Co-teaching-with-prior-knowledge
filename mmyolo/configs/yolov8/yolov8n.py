_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = '/home/user02/mmyolo/data/coco/'
class_name = ('person', 'car', 'ft', 'tlj', 'sl', 'ldc', 'tk', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])


deepen_factor = 0.33
widen_factor = 0.25

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/clear_train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)