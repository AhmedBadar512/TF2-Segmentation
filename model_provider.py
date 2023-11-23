import models
# TODO: Add support for multitask models
__all__ = ['get_model']


_models = {
    'unet': models.UNet,
    'unet_exp': models.UNet_Expanded,
    'deeplabv3': models.Deeplabv3plus,
    'deeplabv3_exp': models.Deeplabv3PlusExpanded,
    'bisenetv2': models.BiSeNetv2
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models and name not in _ganmodels:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
