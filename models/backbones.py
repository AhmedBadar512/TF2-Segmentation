import keras
from models.c_backbones.hardnet import HarDNet


def get_backbone(name="resnet50", **kwargs):
    base_backbone = True
    if name.lower() == "resnet50":
        return keras.applications.ResNet50(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "resnet50v2":
        return keras.applications.ResNet50V2(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "resnet101":
        return keras.applications.ResNet101(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "resnet101v2":
        return keras.applications.ResNet101V2(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "resnet152":
        return keras.applications.ResNet152(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "xception":
        return keras.applications.Xception(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "mobilenetv3large":
        return keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', **kwargs), base_backbone
    elif name.lower() == "mobilenetv3small":
        return keras.applications.MobileNetV3Small(include_top=False, weights='imagenet', **kwargs), base_backbone
    else:
        base_backbone = False
    # ============================= Custom backbones ======================== #
    # These must always return a list of features to be used
    if name.lower() == "hardnet39":
        return HarDNet(arch=39), base_backbone
    elif name.lower() == "hardnet68":
        return HarDNet(arch=68), base_backbone
    elif name.lower() == "hardnet85":
        return HarDNet(arch=85), base_backbone
    else:
        raise ModuleNotFoundError(f'Model "{name}" not yet implemented')


def get_bb_dict(backbone, divisible_factor=1):
    """
    Returns a dict with keys as tuples representing (h, w) from (None, h, w, c) shape
    """
    final_dict = {}
    for n, bb_layer in enumerate(backbone.layers):
        if divisible_factor < bb_layer.output.shape[1] and divisible_factor < bb_layer.output.shape[2]:
            layer_shape = tuple(map(lambda x: round(x / divisible_factor) * divisible_factor, bb_layer.output.shape[1:3]))
        else:
            layer_shape = bb_layer.output.shape[1:3]
        if layer_shape not in final_dict.keys():
            final_dict[layer_shape] = [n]
        else:
            final_dict[layer_shape].append(n)
    return final_dict

def get_custom_bb_dict(outputs, divisible_factor=1):
    """
    Returns a dict with keys as tuples representing (h, w) from (None, h, w, c) shape
    """
    final_dict = {tuple(map(lambda x: round(x / divisible_factor) * divisible_factor if x > divisible_factor else x, out.shape[1:3])): n for n, out in enumerate(outputs)}
    return final_dict