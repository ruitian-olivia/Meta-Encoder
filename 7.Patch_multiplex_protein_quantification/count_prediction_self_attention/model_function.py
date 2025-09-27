import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from attention_utils import MultiheadAttention

class protein_poisson_count(nn.Module):
    def __init__(self, embed_dim, num_heads, encoding_size, num_marker):
        super(protein_poisson_count, self).__init__()

        self.cross_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=encoding_size, vdim=encoding_size, qdim=encoding_size)

        self.head = nn.Linear(embed_dim, num_marker)
        
    def forward(self, features):
        features = features.unsqueeze(0)
        features, _ = self.cross_attention(features, features, features)
        features = features.squeeze()
        
        lambda_params = nn.functional.softplus(self.head(features))
        
        return lambda_params
    
