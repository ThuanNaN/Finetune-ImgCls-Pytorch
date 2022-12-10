

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

