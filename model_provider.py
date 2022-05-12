from models.unet import *
from models.deeplabv3 import *
from models.deeplabv3_exp import *
from models.unet_expanded import *

__all__ = ['get_model']


_models = {
    'unet': UNet,
    'unet_exp': UNet_Expanded,
    'deeplabv3': Deeplabv3plus,
    'deeplabv3_exp': Deeplabv3PlusExpanded
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
