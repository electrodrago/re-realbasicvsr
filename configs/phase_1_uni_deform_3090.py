_base_ = 'default_runtime.py'

experiment_name = 'realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds'
work_dir = f'/workspace/work_dirs/{experiment_name}'
save_dir = '/workspace/work_dirs/'

scale = 4

# model settings
model = dict(
    type='Re_RealBasicVSR',
    generator=dict(
        type='Re_RealBasicVSRNet',
        mid_channels=64, 
        num_blocks=20,
        num_cleaning_blocks=10,
        max_residue_magnitude=10,
        spynet_pretrained='/workspace/re-realbasicvsr/spynet.pth'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    cleaning_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    is_use_sharpened_gt_in_pixel=True,
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='FixedCrop', keys=['gt'], crop_size=(256, 256)),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['img']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.2, 0.7, 0.1],  # up, down, keep
            resize_scale=[0.15, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0],
            resize_step=0.015,
            is_size_even=True),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['img'],
    ),
    dict(
        type='RandomVideoCompression',
        params=dict(
            codec=['h264'],
            codec_prob=[1 / 3.],
            bitrate=[1e4, 1e5]),
        keys=['img'],
    ),
    dict(
        type='RandomBlur',
        params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.3, 0.4, 0.3],  # up, down, keep
            resize_scale=[0.3, 1.2],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            resize_step=0.03,
            is_size_even=True),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['img'],
    ),
    dict(
        type='DegradationsWithShuffle',
        degradations=[
            dict(
                type='RandomVideoCompression',
                params=dict(
                    codec=['h264'],
                    codec_prob=[1 / 3.],
                    bitrate=[1e4, 1e5]),
                keys=['img'],
            ),
            [
                dict(
                    type='RandomResize',
                    params=dict(
                        target_size=(64, 64),
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
                ),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=['sinc'],
                        kernel_prob=[1],
                        omega=[3.1416 / 3, 3.1416],
                        omega_step=0.0628),
                ),
            ]
        ],
        keys=['img'],
    ),
    dict(type='Clip', keys=['img']),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

data_root = 'data'

train_dataloader = dict(
    num_workers=12,
    batch_size=3,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root="/workspace/REDS/train",
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        depth=1,
        num_input_frames=15,
        pipeline=train_pipeline))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100000)

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))))

# NO learning policy

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=500,
        save_optimizer=True,
        out_dir=save_dir,
        max_keep_ckpts=100,
        by_epoch=False),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='BasicVisualizationHook', interval=5),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)