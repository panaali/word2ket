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