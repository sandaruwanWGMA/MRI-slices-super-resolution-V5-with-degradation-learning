import argparse
import torch


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Training options for super-resolution models"
        )
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        # self.parser.add_argument(
        #     "--dataroot", type=str, required=True, help="Path to the dataset directory"
        # )
        self.parser.add_argument(
            "--name",
            type=str,
            default="experiment",
            help="Experiment name for saving logs and models",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of subprocesses to use for data loading",
        )
        self.parser.add_argument(
            "--model_type",
            type=str,
            default="super_resolution_model",
            help="Type of model to train: e.g., 'sr_unet', 'multi_gdn', 'vgg_patch_gan'",
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=8, help="Batch size for training"
        )
        self.parser.add_argument(
            "--epoch_count",
            type=int,
            default=1,
            help="Start counting epochs from this number",
        ),
        self.parser.add_argument(
            "--n_epochs",
            type=int,
            default=10,
            help="Number of epochs at the initial learning rate",
        )
        self.parser.add_argument(
            "--n_epochs_decay",
            type=int,
            default=10,
            help="Number of epochs to linearly decay the learning rate to zero",
        )
        self.parser.add_argument(
            "--continue_train",
            action="store_true",
            help="Continue training from the last saved epoch",
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints/",
            help="Directory to save model checkpoints",
        )
        self.parser.add_argument(
            "--checkpoint_dir_vol",
            type=str,
            default="./checkpoints/models_per_each_mri_volume",
            help="Directory to save model checkpoints",
        )
        self.parser.add_argument(
            "--checkpoint_dir_epoch",
            type=str,
            default="./checkpoints/models_per_epoch",
            help="Directory to save model checkpoints",
        )
        self.parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="Epoch to start resuming training ('latest' or specific epoch number)",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.0002,
            help="Initial learning rate for Adam optimizer",
        )
        self.parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="Comma-separated GPU IDs (e.g., '0,1,2') for training; '-1' for CPU",
        )
        self.parser.add_argument(
            "--print_freq",
            type=int,
            default=2,
            help="Frequency of printing training results to the console",
        )
        self.parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=10,
            help="Frequency of saving the latest results during training",
        )
        self.parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=5,
            help="Frequency of saving checkpoints at the end of specified number of epochs",
        )
        self.parser.add_argument(
            "--display_freq",
            type=int,
            default=2,
            help="Frequency of displaying results on the training console",
        )
        # options specific to super_resolution_model components
        self.parser.add_argument(
            "--image_size",
            type=int,
            default=256,
            help="Size of the input and output images (assumes square images)",
        )
        self.parser.add_argument(
            "--in_channels",
            type=int,
            default=1,
            help="Number of input channels (e.g., 3 for RGB images)",
        )
        self.parser.add_argument(
            "--out_channels",
            type=int,
            default=1,
            help="Number of output channels (e.g., 3 for RGB images)",
        )
        self.parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            help="Freeze encoder layers of the SRUNet model",
        )
        self.parser.add_argument(
            "--patch_size",
            type=int,
            default=70,
            help="Patch size for VGGStylePatchGAN model",
        )
        self.parser.add_argument(
            "--unfreeze_layers",
            type=int,
            default=["blocks.3", "blocks.4", "blocks.5", "blocks.6"],
            help="Unfreezed layers for SRUNet",
        )
        # Parameters for loss functions
        self.parser.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="Weight for perceptual quality loss",
        )
        self.parser.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="Weight for feature matching loss in perceptual loss",
        )
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=1.0,
            help="Weight for style loss component in perceptual loss",
        )
        self.parser.add_argument(
            "--delta",
            type=float,
            default=1.0,
            help="Weight for adversarial loss in perceptual_adversarial_loss",
        )
        self.parser.add_argument(
            "--lambda_tv",
            type=float,
            default=1.0,
            help="Weight for total variation loss in GDNLoss",
        )
        # Options specific to CustomDeepLab model
        self.parser.add_argument(
            "--num_classes",
            type=int,
            default=1,
            help="Number of output classes for segmentation",
        )
        self.parser.add_argument(
            "--freeze_backbone",
            default=True,
            help="Freeze the backbone of the CustomDeepLab model",
        )
        self.parser.add_argument(
            "--freeze_classifier",
            default=False,
            help="Freeze the classifier of the CustomDeepLab model",
        )

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Determine the device to use based on --gpu_ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = [int(id) for id in str_ids if int(id) >= 0]  # Filter out -1 (CPU)
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
            torch.cuda.set_device(opt.gpu_ids[0])  # Set the first GPU as the default
        else:
            opt.device = torch.device("cpu")

        self.print_options(opt)
        return opt

    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            # Convert opt.device to a string if it's a torch.device object
            if isinstance(v, torch.device):
                v = str(v)
            if isinstance(v, list):  # Convert list to string to avoid formatting errors
                v = ", ".join(map(str, v))
            default = self.parser.get_default(k)
            comment = f"\t[default: {default}]" if v != default else ""
            message += f"{k:>25}: {v:<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)
