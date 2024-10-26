_base_ = './yolov6_n_syncbn_fast_8xb32-300e_coco.py'

data_root = '/home/user02/mmyolo/data/coco/'
class_name = ('person', 'Forklift', 'Tractor', 'Shovel loader', 'Furnace bottom car')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

max_epochs = 300
train_batch_size_per_gpu = 16
train_num_workers = 4
num_last_epochs = 5


model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)))

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

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - num_last_epochs

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa