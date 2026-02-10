import torch.nn as nn
from .modules import GraphAttentionLayer
from torch_geometric.nn import GATConv

class PyGGAT(nn.Module):
    def __init__(self,node_dim:int,latent_dim:int=32,num_heads:int=3,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gat_block=nn.Sequential(
            GATConv(in_channels=node_dim,out_channels=latent_dim,heads=num_heads,concat=True),
            nn.ReLU(),
            GATConv(in_channels=num_heads*latent_dim,out_channels=latent_dim,heads=num_heads,concat=False)
        )
        self.decoder=nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=2*latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=1)
        )

    def forward(self,x,edge_index):
        """
        x: [N,node_dim]
        edge_index: [2,E]
        """
        h=self.gat_block(x,edge_index)
        z=self.decoder(h)
        return z

class CustomGAT(nn.Module):
    def __init__(self,node_dim:int,latent_dim:int=32,num_heads:int=3,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gat_block=nn.Sequential(
            GraphAttentionLayer(node_dim=node_dim,latent_dim=latent_dim,num_heads=num_heads,is_last=False),
            nn.ReLU(),
            GraphAttentionLayer(node_dim=num_heads*latent_dim,latent_dim=latent_dim,num_heads=num_heads,is_last=True)
        )
        self.decoder=nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=2*latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=1)
        )

    def forward(self,x,edge_index):
        """
        x: [N,node_dim]
        edge_index: [2,E]
        """
        h=self.gat_block(x,edge_index)
        z=self.decoder(h)
        return z