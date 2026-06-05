import torch
import torch.nn as nn
from modules import GraphAttentionEmbedding

class GAT(nn.Module):
    def __init__(self,
            node_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim

        # module
        self.gat_layer_1=GraphAttentionEmbedding(
            node_dim=node_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            n_head=3,
            is_concat=True
        )
        self.gat_layer_2=GraphAttentionEmbedding(
            node_dim=output_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            n_head=3,
            is_concat=False
        )
        self.relu=nn.ReLU()

    def forward(self):
        return NotImplemented

class GAT_Link_Prediction(GAT):
    def __init__(self,
            node_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super(GAT_Link_Prediction,self).__init__(
            node_dim,
            latent_dim,
            output_dim
        )