from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    if opt["lr_policy"] == "linear":

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt["epoch_count"] - opt["n_epochs"]) / float(
                opt["n_epochs_decay"] + 1
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt["lr_policy"] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt["lr_decay_iters"], gamma=0.1
        )

    elif opt["lr_policy"] == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )

    elif opt["lr_policy"] == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt["n_epochs"], eta_min=0
        )

    else:
        raise NotImplementedError(
            f"Learning rate policy {opt['lr_policy']} is not implemented"
        )
    return scheduler
