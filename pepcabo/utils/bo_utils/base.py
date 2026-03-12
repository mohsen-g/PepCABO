import torch
from collections import OrderedDict
from torch import nn
class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish,dropout_prob):
        if swish:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("swish", Swish()),
                ('norm',nn.LayerNorm(output_dim))
                #('dropout', torch.nn.Dropout(dropout_prob))
            ]))

class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims,dropout_prob=0.1, swish=True):
        prev_dims = [input_dim] + list(hidden_dims[:-1])
        layers = OrderedDict([
            (f"hidden{i + 1}", _LinearBlock(prev_dim, current_dim, swish=swish,dropout_prob=dropout_prob))
            for i, (prev_dim, current_dim) in enumerate(zip(prev_dims, hidden_dims))
        ])
        self.output_dim = hidden_dims[-1]

        super().__init__(layers)

class HierarchicalAttentionFeatureExtractor(nn.Module):
    def __init__(self, hidden_dims, num_heads=4, peptide_dim=64, allele_dim=64):
        super().__init__()
        self.peptide_dim = peptide_dim
        self.allele_dim = allele_dim
        

        self.peptide_self_attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )

        self.peptide_encoder = nn.Sequential(
            nn.Linear(peptide_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 512)
        )

        self.allele_self_attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )
        self.allele_encoder = nn.Sequential(
            nn.Linear(peptide_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 512)
        )
        #self.proj_to_common = nn.Linear(peptide_dim + allele_dim, hidden_dims[0])
        
        self.cross_attention_allele_peptide = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )

        self.cross_attention_peptide_allele = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )
        
        #self.output_proj = nn.Linear(peptide_dim+allele_dim, hidden_dims[-1])
        self.output_proj = nn.Sequential(
            nn.Linear(1024, 128),
            nn.SiLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        # Split features
        peptide_features = x[..., :self.peptide_dim]
        allele_features = x[..., self.peptide_dim:]
        
        peptide_self = self.peptide_encoder(
            peptide_features
        ).reshape(peptide_features.shape[0],16,32)

        allele_self = self.allele_encoder(
            allele_features
        ).reshape(allele_features.shape[0],16,32)
        

        peptide_attended, _ = self.cross_attention_allele_peptide(
            query=peptide_self, key=allele_self, value=allele_self
        )
        
        allele_attended, _ = self.cross_attention_peptide_allele(
            query=allele_self, key=peptide_self, value=peptide_self
        )

        
        # Step 2: Combine and project to common space
        combined = torch.cat([allele_attended.reshape(allele_features.shape[0],16*32),
                              peptide_attended.reshape(allele_features.shape[0],16*32)], dim=-1)
        common_proj = self.output_proj(combined)#.unsqueeze(1)
        
        
        return common_proj