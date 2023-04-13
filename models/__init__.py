from models.BranchedERFNet import BranchedERFNet,BranchedHyperNet,BranchedMultiModalNet
def get_model(name, model_opts):
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    if name == "branched_hypernet":
        model = BranchedHyperNet(**model_opts)
        return model
    if name == "branched_multimodalnet":
        model = BranchedMultiModalNet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))