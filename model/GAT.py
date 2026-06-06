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
        self.encoder=nn.Sequential(
            GraphAttentionEmbedding(
                node_dim=node_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                n_head=3,
                is_concat=True
            ),
            nn.ReLU(),
            GraphAttentionEmbedding(
                node_dim=output_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                n_head=3,
                is_concat=False
            )
        )

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
        # module
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=output_dim+output_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )
    def forward(self,
            node_ft:torch.Tensor,
            edge_index:torch.Tensor
        ):
        """
        """
        h=self.encoder(node_ft=node_ft,edge_index=edge_index)
        z=self.decoder()