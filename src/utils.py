def round_ms(seconds):
    return round(1000 * seconds, 3)


def get_image_size(model):
    if model in {"alexnet", "resnet", "vgg"}:
        image_size = 224
    elif model in {"inception"}:
        image_size = 299
    else:
        raise RuntimeError("Image size not defined for model: " + str(model))

    return image_size