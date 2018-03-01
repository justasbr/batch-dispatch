import random
def get_random_num(raw_img):
    return random.randint(0, 1000)

def get_inference_handle(model, batch_size):
    print("Batch size", batch_size) 
    if model not in {"alexnet","vgg","resnet","inception"}:
        raise Exception("TRT did not have model:" +model)
    if model == "alexnet":
        from tensorRT.alexnet import infer
    elif model == "vgg":
        from tensorRT.vgg import infer
    elif model == "resnet":
        from tensorRT.resnet import infer
    elif model == "inception":
        from tensorRT.inception import infer
    if infer is not None:
        return infer
    else:
        raise Exception("Something did not work - TRT, model:" + model)
