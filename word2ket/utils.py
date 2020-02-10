import torch
from .EmbeddingKet import EmbeddingKet, EmbeddingKetXS
import logging

def summary(model):
    result = []
    total_params_element = 0
    def check_trainable(module):
        nonlocal total_params_element
        if len(list(module.children())) == 0:
            num_param = 0
            num_trainable_param = 0
            num_param_numel = 0
            for parameter in module.parameters():
                num_param += 1
                if parameter.requires_grad:
                    num_param_numel += parameter.numel()
                    total_params_element += parameter.numel()
                    num_trainable_param += 1

            result.append({'module': module, 'num_param': num_param , 'num_trainable_param' : num_trainable_param, 'num_param_numel': num_param_numel})
    model.apply(check_trainable)
    
    print("{: <85} {: <17} {:,<20} {: <40}".format('Module Name', 'Total Parameters', 'Trainable Parameters', '# Elements in Trainable Parametrs'))
    for row in result:
        print("{: <85} {: <17} {: <20} {: <40,}".format(row['module'].__str__(), row['num_param'], row['num_trainable_param'], row['num_param_numel']))
    print('Total number of trainable parameters elements {:,}'.format(total_params_element))
    return total_params_element

def ketify(model, order, rank, use_EmbeddingKetXS=True):
    """
    This function replaces all nn.Embedding layers of the model
    with EmbeddingKet layer.
    The input model should be a torch.nn.Module inherited object.
    """
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Embedding):
            if use_EmbeddingKetXS:
                ketLayer = EmbeddingKetXS(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    order = order,
                    rank = rank
                )
            else:
                ketLayer = EmbeddingKet(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    order = order,
                    rank = rank
                )
            # put the layer on gpu if it was on gpu
            if next(child.parameters()).is_cuda: ketLayer.cuda()
            setattr(model, name, ketLayer)
            logging.info("Replaced " + name + " in " + model.__class__.__name__)
        else:
            ketify(child, order, rank)
    return model