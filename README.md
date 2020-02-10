# word2ket
**word2ket** is an space-efficient embedding layer that can reduce the space required to store the embeddings by up to 100,000x.

This is a PyTorch implementaion of the embedding layer that is proposed in the paper [word2ket: Space-efficient Word Embeddings inspired by Quantum Entanglement](https://arxiv.org/abs/1911.04975). 

- [word2ket](#word2ket)
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Install word2ket](#install-word2ket)
- [Getting Started](#getting-started)
- [Running examples](#running-examples)
  - [Dependencies](#dependencies-1)
  - [Download Datasets](#download-datasets)
  - [Run text summarization](#run-text-summarization)
  - [Run German-English machine translation](#run-german-english-machine-translation)
- [Reference](#reference)
  - [APA](#apa)
  - [BibTex](#bibtex)
- [License](#license)

# Installation
## Dependencies
```bash
# Install PyTorch
conda install pytorch torchvision -c pytorch
# Install GPyTorch
conda install gpytorch -c gpytorch
```
## Install word2ket
```bash
pip install word2ket
```
# Getting Started
You can directly use `EmbeddingKet` and `EmbeddingKetXS` layers in your model definition or you can use the `ketify` function to automatically replace all the `nn.Embedding` layers in your model. Below you can see a model that is using `7,680,000` parameters for its embedding could be `ketify` to just use `448` parameters.

```python
# examples/demo.py 
from word2ket import EmbeddingKet, EmbeddingKetXS, ketify, summary
from torch import nn
import logging
logging.basicConfig(level=logging.INFO)


# Word2Ket Embedding Layer
w2v_embedding_layer = EmbeddingKetXS(num_embeddings=30000, embedding_dim=256, order=4, rank=1)
summary(w2v_embedding_layer)
"""
INFO:root:EmbeddingKetXS num_embeddings_leaf: 14
INFO:root:EmbeddingKetXS embedding_dim_leaf: 4
INFO:root:EmbeddingKetXS weight_leafs shape: torch.Size([4, 1, 14, 4])
Module Name                                                                           Total Parameters  Trainable Parameters # Elements in Trainable Parametrs       
EmbeddingKetXS(30000, 256)                                                            1                 1                    224                                     
Total number of trainable parameters elements 224
"""

# PyTorch Embedding Layer
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    
    def forward(self, x):
        self.embedding()
model = MyModel(vocab_size=30000, embedding_dim=256)
print("embedding.weight.shape: ", model.embedding.weight.shape)
"""
embedding.weight.shape:  torch.Size([30000, 256])
"""
summary(model)
"""
Module Name                                                                           Total Parameters  Trainable Parameters # Elements in Trainable Parametrs       
Embedding(30000, 256)                                                                 1                 1                    7,680,000                               
Total number of trainable parameters elements 7,680,000
"""

# Replace the nn.Embedding to EmbeddingKetXS automatically using the ketify function.
ketify(model, order=4, rank=2, use_EmbeddingKetXS=True)
summary(model)
"""
INFO:root:EmbeddingKetXS num_embeddings_leaf: 14
INFO:root:EmbeddingKetXS embedding_dim_leaf: 4
INFO:root:EmbeddingKetXS weight_leafs shape: torch.Size([4, 2, 14, 4])
INFO:root:Replaced embedding in MyModel
Module Name                                                                           Total Parameters  Trainable Parameters # Elements in Trainable Parametrs       
EmbeddingKetXS(30000, 256)                                                            1                 1                    448                                     
Total number of trainable parameters elements 448
"""
```

# Running examples
## Dependencies
```bash
# Install Texar-PyTorch
pip install texar-pytorch
# Install Rouge
pip install rouge
```

## Download Datasets
```bash
cd ./examples/texar/
python ./prepare_data.py --data iwslt14
python ./prepare_data.py --data giga
```

## Run text summarization
Using **GIGAWORD** dataset.
```bash
# Using Pytorch nn.Embedding Layer
python seq2seq_attn.py --embedding_type nn.Embedding --gpu 4 --runName G_000 --config-model config_model --config-data config_giga

# Using EmbeddingKet Layer
python seq2seq_attn.py --embedding_type EmbeddingKet   --gpu 0 --runName V2K_G_000       --config-model config_model --config-data config_giga --order 4 --rank 1

# Using EmbeddingKetXS Layer
python seq2seq_attn.py --embedding_type EmbeddingKetXS --gpu 0 --runName V2K_XS_G_000 --config-model config_model --config-data config_giga --order 4 --rank 1
```

## Run German-English machine translation
Using **IWSLT2014** (DE-EN) dataset
```bash
# Using Pytorch nn.Embedding Layer
python seq2seq_attn.py --embedding_type nn.Embedding --gpu 4 --runName I_000 --config-model config_model --config-data config_iwslt14

# Using EmbeddingKet Layer
python seq2seq_attn.py --embedding_type EmbeddingKet   --gpu 0 --runName V2K_I_000       --config-model config_model --config-data config_iwslt14 --order 4 --rank 1

# Using EmbeddingKetXS Layer
python seq2seq_attn.py --embedding_type EmbeddingKetXS --gpu 0 --runName V2K_XS_I_000 --config-model config_model --config-data config_iwslt14 --order 4 --rank 1

```


# Reference
If you use **word2ket**, please cite the paper:

## APA
```
Panahi, A., Saeedi, S., & Arodz, T. (2020). word2ket: Space-efficient Word Embeddings inspired by Quantum Entanglement. In proceedings of the International Conference on Learning Representations (ICLR) 2020
```

## BibTex
```
@inproceedings{panahi2020wordket,
  title={word2ket: Space-efficient Word Embeddings inspired by Quantum Entanglement},
  author={Aliakbar Panahi and Seyran Saeedi and Tom Arodz},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=HkxARkrFwB}
}
```
# License
[BSD-3-Clause](./LICENSE)


