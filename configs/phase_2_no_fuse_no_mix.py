_base_ = './phase_1_bi_deform_1x_A100_40.py'

experiment_name = 'no_fuse_no_mix'
work_dir = f'/content/drive/MyDrive/1THESIS/bi_deform_no_fuse_no_mix/{experiment_name}'
save_dir = '/content/drive/MyDrive/1THESIS/bi_deform_no_fuse_no_mix/'

load_from = '/content/drive/MyDrive/1THESIS/bi_deform_final/phase_2/iter_250000.pth'  # noqa

scale = 4

# model settings
model = dict(
    type='Re_RealBasicVSR',
    generator=dict(
        type='Re_RealBasicVSRNet',
        mid_channels=64, 
        num_blocks=12,
        num_cleaning_blocks=15,
        max_residue_magnitude=10,
        spynet_pretrained='/content/re-realbasicvsr/spynet.pth'),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-2,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5, betas=(0.9, 0.99))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=150000, val_interval=510000)