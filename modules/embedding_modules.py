import torch
import torch.nn as nn
from utils import ModuleUtils

class GraphAttentionEmbedding(nn.Module):
    def __init__(self,
            node_dim:int,
            latent_dim:int,
            output_dim:int,
            n_head:int=1,
            is_concat:bool=True
        ):
        super().__init__()
        self.node_dim=node_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.n_head=n_head
        self.is_concat=is_concat

        # module
        self.linear=nn.Linear(node_dim,n_head*latent_dim)
        self.attn_src=nn.Parameter(torch.empty(1,n_head,latent_dim))
        self.attn_tar=nn.Parameter(torch.empty(1,n_head,latent_dim))
        if is_concat:
            self.output_linear=nn.Linear(n_head*latent_dim,output_dim)
        else:
            self.output_linear=nn.Linear(latent_dim,output_dim)
        self.leakly_relu=nn.LeakyReLU(negative_slope=0.2)

    def forward(self,
            node_ft:torch.Tensor,
            edge_index:torch.Tensor
        ):
        """
        Input:
            node_ft: [N,node_dim]
            edge_index: [2,E]
        """
        N=node_ft.size(0)
        src,tar=edge_index

        # 1. node feature projection
        node_ft=self.linear(node_ft).view(N,self.n_head,self.latent_dim) # [N,n_head,latent_dim]

        # 2. node-level attention coefficients
        coef_src=(node_ft*self.attn_src).sum(dim=-1) # [N,n_head]
        coef_tar=(node_ft*self.attn_tar).sum(dim=-1) # [N,n_head]

        # 3. edge-level attention coefficients
        coef_edge=coef_src[src]+coef_tar[tar] # [E,n_head]

        # 4. apply LeaklyReLU
        coef_edge=self.leakly_relu(coef_edge)

        # 5. target node별 softmax
        alpha_edge=ModuleUtils.group_softmax(coef_edge,tar,N)

        # 6. msg 계산
        msg=node_ft[src]*alpha_edge.unsqueeze(-1) # node_ft[src]: [E,n_head,latent_dim], alpha: [E,n_head,1], msg: [E,n_head,latent_dim]

        # 7. target node 별 aggregation
        output=torch.zeros(
            (N,self.n_head,self.latent_dim),
            device=node_ft.device,
            dtype=node_ft.dtype
        )
        output.index_add_(dim=0,index=tar,source=msg) # [N,n_head,latent_dim]

        # 8. multi-head concat or mean
        if self.is_concat:
            output=output.reshape(N,self.n_head*self.latent_dim) # [N,n_head*latent_dim]
        else:
            output=output.mean(dim=1) # [N,latent_dim]
        
        # 9. output projection
        output=self.output_linear(output) # [N,output_dim]
        return output