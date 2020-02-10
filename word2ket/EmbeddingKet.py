import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
try:
    from gpytorch.lazy import KroneckerProductLazyTensor, NonLazyTensor
except:
    raise Exception("You should install GPyTorch. \n"
                    + "$ conda install gpytorch -c gpytorch \n"
                    + "https://github.com/cornellius-gp/gpytorch \n"
                    )

class EmbeddingKet(nn.Embedding):
    r"""This is a new embedding using Kronecker products.
    Order = order + 1
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'order', 'rank', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, order = 1, rank = 1, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        """
            order: number of times we do the tensor product
            rank : Rank of the matrix, the dimension that we calcualte the sum of batches
        """
        super(nn.Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_dim_leaf = math.ceil((embedding_dim)**(1/order))
        logging.info('EmbeddingKet base num_embeddings: ' + str(self.num_embeddings))
        logging.info('EmbeddingKet embedding_dim_leaf: ' + str(self.embedding_dim_leaf))
        self.rank = rank
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.order = order
        if _weight is None:
            # Ali: Creating Leaf Weights for Tensor product
            self.weight_leafs = nn.Parameter(torch.Tensor(  # TODO:  During backward do we need all of these parameters to be loaded ? can we do better?
                self.order, self.rank,  self.num_embeddings, self.embedding_dim_leaf))
            logging.info('EmbeddingKet weight_leafs shape: ' + str(self.weight_leafs.shape))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [self.order, self.rank, self.num_embeddings_leaf, self.embedding_dim_leaf], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight_leafs = nn.Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_leafs)
        # if self.padding_idx is not None:
        #     with torch.no_grad():
        #         self.weight_leafs[self.padding_idx].fill_(0)

    def forward(self, input):
        w = self.weight_leafs
        input_1d = input
        if input.dim() == 2:
            input_1d = input.contiguous().view(1,-1)
        # TODO: grab the input from the w and perform all the operations just on that
        # w = w[:,:,input_1d,:]  #DEVICE ASSERTION ERROR WHEN DOING MSE !?!
        # w = nn.LayerNorm(w.shape[-1]).cuda()(w) # not used in experiments before.
        if self.order == 2:
            w01 = (w[0,:,:,:,None] * w[1,:,:,None,:] )
            w01 = w01.view(self.rank,  self.num_embeddings, -1)
            # w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            
            weight = w01.sum(0)
        elif self.order == 4:
            w01 = (w[0,:,:,:,None] * w[1,:,:,None,:] )
            w01 = w01.view(self.rank,  self.num_embeddings, -1)
            # w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)

            w23 = (w[2,:,:,:,None] * w[3,:,:,None,:] )
            w23 = w23.view(self.rank,  self.num_embeddings, -1)
            # w23 = nn.LayerNorm(w23.shape[-2:]).cuda()(w23)

            w0123 = (w01[:,:,:,None] * w23[:,:,None,:] )
            w0123 = w0123.view(self.rank,  self.num_embeddings, -1)
            # w0123 = nn.LayerNorm(w0123.shape[-2:]).cuda()(w0123)

            weight = w0123.sum(0)
        elif self.order == 8:
            w01 = (w[0,:,:,:,None] * w[1,:,:,None,:] )
            w01 = w01.view(self.rank,  self.num_embeddings, -1)
            w23 = (w[2,:,:,:,None] * w[3,:,:,None,:] )
            w23 = w23.view(self.rank,  self.num_embeddings, -1)
            w45 = (w[4,:,:,:,None] * w[5,:,:,None,:] )
            w45 = w45.view(self.rank,  self.num_embeddings, -1)
            w67 = (w[6,:,:,:,None] * w[7,:,:,None,:] )
            w67 = w67.view(self.rank,  self.num_embeddings, -1)
            w0123 = (w01[:,:,:,None] * w23[:,:,None,:] )
            w0123 = w0123.view(self.rank,  self.num_embeddings, -1)
            w4567 = (w45[:,:,:,None] * w67[:,:,None,:] )
            w4567 = w4567.view(self.rank,  self.num_embeddings, -1)
            w01234567 = (w0123[:,:,:,None] * w4567[:,:,None,:] )
            w01234567 = w01234567.view(self.rank,  self.num_embeddings, -1)
            weight = w01234567.sum(0)
        else:
            raise Exception(f'The order {self.order} is not yet implemented')
        
        weight = weight[:,:self.embedding_dim]
        
        return F.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)

    def knocker_product(self, a, b):
        res = []
        for i in range(a.size(-2)):
            row_res = []
            for j in range(a.size(-1)):
                row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
            res.append(torch.cat(row_res, -1))
        return torch.cat(res, -2)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False): 
        assert embeddings.dim() == 4, \
            'Embeddings parameter is expected to be 4-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding


class EmbeddingKetXS(nn.Embedding):
    r"""This is a new embedding using Kronecker products.
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'order', 'rank', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, order = 1, rank = 1, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, lazy=True):
        """
            order: number of times we do the tensor product
            rank : Rank of the matrix, the dimension that we calcualte the sum of batches
        """
        super(nn.Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_leaf = math.ceil((num_embeddings)**(1/order))
        self.embedding_dim_leaf = math.ceil((embedding_dim)**(1/order))
        logging.info('EmbeddingKetXS num_embeddings_leaf: ' + str(self.num_embeddings_leaf))
        logging.info('EmbeddingKetXS embedding_dim_leaf: ' + str(self.embedding_dim_leaf))
        self.rank = rank
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.order = order
        if _weight is None:
            # Creating Leaf Weights for Kronecker product
            self.weight_leafs = nn.Parameter(torch.Tensor(
                self.order, self.rank,  self.num_embeddings_leaf, self.embedding_dim_leaf))
            logging.info('EmbeddingKetXS weight_leafs shape: ' + str(self.weight_leafs.shape))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [self.order, self.rank, self.num_embeddings_leaf, self.embedding_dim_leaf], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight_leafs = nn.Parameter(_weight)
        self.sparse = sparse
        self.lazy = lazy

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight_leafs)
        torch.nn.init.normal_(self.weight_leafs)
        # if self.padding_idx is not None:  # I can't set the value of a specific location to zero since I don't know it's final location.
        #     with torch.no_grad():
        #         self.weight_leafs[self.padding_idx].fill_(0)

    def forward(self, input):
        # Here I should calculate the (final) weight first using tensor products and the rest is exactly the same
        # w = nn.BatchNorm2d(self.weight_leafs.shape[1]).cuda()(self.weight_leafs)
        w = self.weight_leafs
        if self.lazy:
            self.weight = KroneckerProductLazyTensor(*NonLazyTensor(w)).sum(dim=0) # get the sum of the batch of product 
            logging.debug('self.weight.shape: ' + str(self.weight.shape))
            if input.dim() == 1: #
                return self.weight[input].base_lazy_tensor.evaluate().sum(dim=-3)[:,:self.embedding_dim] # https://github.com/cornellius-gp/gpytorch/pull/871
            elif input.dim() == 2:
                input_1d = input.contiguous().view(1,-1)
                result = self.weight[input_1d[0]].base_lazy_tensor.evaluate().sum(dim=-3)[:,:self.embedding_dim] #TODO: Not sure if this selection (self.embedding_dim) is correct in here. # https://github.com/cornellius-gp/gpytorch/pull/871
                return result.view(input.shape[0], input.shape[1], -1)
            else:
                raise Exception('This input dimesion is not yet implemented')
        else:
            weight_leafs_product = w[0]
            for i in range(1, self.order):
                weight_leafs_product = self.knocker_product(weight_leafs_product, w[i])
            self.weight = weight_leafs_product.sum(dim=0)


            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)

    def knocker_product(self, a, b):
        res = []
        for i in range(a.size(-2)):
            row_res = []
            for j in range(a.size(-1)):
                row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
            res.append(torch.cat(row_res, -1))
        return torch.cat(res, -2)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False): 
        assert embeddings.dim() == 4, \
            'Embeddings parameter is expected to be 4-dimensional'
        order, rank, rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding
