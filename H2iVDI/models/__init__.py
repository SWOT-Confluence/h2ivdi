from .low_froude_model import LowFroudeModel
from .swst3lfb_model import SWST3LFBModel
from .dassflow1dst31_model import DassFlow1DST31BModel

def new_model(model_id: str, data, **kwargs):

    if model_id == "lowfroude":
        model_kwargs = {key:kwargs[key] for key in kwargs if key in ["h0", "h1", "k0", "k1"]}
        model = LowFroudeModel(data, **model_kwargs)
    elif model_id == "swst3lfb":
        model_kwargs = {key:kwargs[key] for key in kwargs if key in ["h0", "h1", "k0", "k1", "bathy_model"]}
        model = SWST3LFBModel(data, **model_kwargs)
    elif model_id == "dassflow1dst31":
        model_kwargs = {key:kwargs[key] for key in kwargs if key in ["h0", "h1", "k0", "k1", "bathy_model"]}
        model = DassFlow1DST31BModel(data, kch=30.0, **model_kwargs)
    else:
        raise ValueError("Wrong model ID: %s" % model_id)

    return model
