import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops,add_self_loops,softmax

class GraphAttentionLayer(nn.Module):
    def __init__(self,node_dim:int,latent_dim:int=32,num_heads:int=3,is_last:bool=True,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.linear=nn.Linear(in_features=node_dim,out_features=num_heads*latent_dim)
        self.attention_linear=nn.Linear(in_features=2*latent_dim,out_features=1)
        self.leaky_relu=nn.LeakyReLU(negative_slope=0.2)
        self.num_heads=num_heads
        self.latent_dim=latent_dim
        self.is_last=is_last

    def forward(self,x,edge_index):
        """
        x: [N,node_dim]
        edge_index: [2,E]
        """
        num_nodes=x.size(0)
        
        edge_index,_=remove_self_loops(edge_index)
        edge_index,_=add_self_loops(edge_index,num_nodes=num_nodes)

        num_edges=edge_index.size(1)

        src_idx,dst_idx=edge_index
        x_src=x[src_idx]
        x_dst=x[dst_idx]

        h_src=self.linear(x_src).view(num_edges,self.num_heads,self.latent_dim) # [E,num_heads,latent_dim]
        h_dst=self.linear(x_dst).view(num_edges,self.num_heads,self.latent_dim) # [E,num_heads,latent_dim]

        attn_input=torch.cat([h_dst,h_src],dim=-1) # [E,num_heads,2*latent_dim]
        e_ij=self.leaky_relu(self.attention_linear(attn_input)) # [E,num_heads,1]
        e_ij=e_ij.squeeze(-1) # [E,num_heads]

        alpha_ij=softmax(e_ij,dst_idx) # [E,num_heads]
        m_ij=alpha_ij.unsqueeze(-1)*h_src # [E,num_heads,1] X [E,num_heads,latent_dim] = [E,num_heads,latent_dim]

        updated_h=scatter(
            m_ij, # [E,num_heads,latent_dim] (edge message)
            dst_idx, # [E] (어디로 갈지)
            dim=0,
            dim_size=num_nodes,
            reduce="sum"
        ) # [N,num_heads,latent_dim]

        if self.is_last:
            updated_h=updated_h.mean(dim=1) # [N,latent_dim]
        else:
            updated_h=updated_h.view(num_nodes,self.num_heads*self.latent_dim) # [N,num_heads*latent_dim]
        return updated_h