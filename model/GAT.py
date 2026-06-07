import torch
import torch.nn as nn
from modules import GraphAttentionEmbedding

class GAT_Base(nn.Module):
    def __init__(self,
            node_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim

        # encoder module
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

    def forward(self):
        return NotImplemented

class GAT_Link_Prediction(GAT_Base):
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
            embed_edge_index:torch.Tensor,
            target_edge_index:torch.Tensor
        ):
        """
        """
        # encoder
        h_1=self.gat_layer_1(node_ft=node_ft,edge_index=embed_edge_index)
        h_1=nn.ReLU(h_1)
        h=self.gat_layer_2(node_ft=node_ft,edge_index=embed_edge_index)

        # decoder
        src,tar=target_edge_index
        src_ft=h[src]
        tar_ft=h[tar]
        edge_ft=torch.cat(
            [src_ft,tar_ft],
            dim=-1
        )
        pred_target_edge_logit=self.decoder(edge_ft) # [target_E,1]
        return pred_target_edge_logit