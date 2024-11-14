_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vd_contrast.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(num_classes=3, loss_decode=dict(type='HieraTripletLossCityscape', num_classes=3, loss_weight=1.0)),
             auxiliary_head=dict(num_classes=3),
             test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=3))

