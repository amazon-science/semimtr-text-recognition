from imgaug import augmenters as iaa


def get_augmentation_pipeline(augmentation_severity=1):
    """
    Defining the augmentation pipeline for SemiMTR pre-training and fine-tuning.
    :param augmentation_severity:
        a. 0 - ABINet augmentation pipeline
        b. 1 - SemiMTR augmentation pipeline
    :return: augmentation_pipeline
    """
    if augmentation_severity == 1:
        augmentations = iaa.Sequential([
            iaa.Invert(0.5),
            iaa.OneOf([
                iaa.ChannelShuffle(0.35),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.KMeansColorQuantization(),
                iaa.HistogramEqualization(),
                iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                iaa.ChangeColorTemperature((1100, 10000))
            ]),
            iaa.OneOf([
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                iaa.OneOf([
                    iaa.GaussianBlur((0.5, 1.5)),
                    iaa.AverageBlur(k=(2, 6)),
                    iaa.MedianBlur(k=(3, 7)),
                    iaa.MotionBlur(k=5)
                ])
            ]),
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.ImpulseNoise(0.1),
                iaa.MultiplyElementwise((0.5, 1.5))
            ])
        ])
    else:
        raise NotImplementedError(f'augmentation_severity={augmentation_severity} is not supported')

    return augmentations
