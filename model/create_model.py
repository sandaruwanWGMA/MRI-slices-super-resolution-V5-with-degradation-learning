from .SRUNet import SRUNet
from .degradation_network import DegradationNetwork
from .VGGStylePatchGAN import VGGStylePatchGAN
from .SuperResolutionModel import SuperResolutionModel
from .CustomDeepLab import CustomDeepLab


def create_model(opt):
    if opt.model_type == "super_resolution_model":
        return SuperResolutionModel(opt)
    elif opt.model_type == "sr_unet":
        return SRUNet(
            image_size=opt.image_size,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            freeze_encoder=opt.freeze_encoder,
        )
    elif opt.model_type == "deeplab":
        return CustomDeepLab(
            in_channels=opt.in_channels,
            num_classes=opt.num_classes,
            freeze_backbone=opt.freeze_backbone,
            freeze_classifier=opt.freeze_classifier,
        )
    elif opt.model_type == "multi_gdn":
        return DegradationNetwork(image_size=opt.image_size)
    elif opt.model_type == "vgg_patch_gan":
        return VGGStylePatchGAN(patch_size=opt.patch_size)
    else:
        raise ValueError(f"Unknown model type: {opt.model_type}")
