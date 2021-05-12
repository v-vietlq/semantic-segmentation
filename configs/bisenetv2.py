cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=2,
    lr_start=1e-2,
    max_iter=80000,
    num_aux_heads = 2,
    weight_decay=5e-4,
    warmup_iters=1000,
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    ims_per_gpu=4,
)