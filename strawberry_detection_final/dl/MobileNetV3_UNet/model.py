import segmentation_models_pytorch as smp

def get_model():
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )