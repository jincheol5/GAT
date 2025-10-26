import torch
import torch.nn as nn
from typing_extensions import Literal
from torch_scatter import scatter_softmax,scatter
from torch_geometric.nn import MessagePassing,GATConv,global_mean_pool

class Custom_GAT_layer(MessagePassing):
    def __init__(self,node_dim,latent_dim,aggr=None,num_head=1,is_final_layer=True):
        super().__init__(aggr=aggr)
        self.initial_linear=nn.Linear(in_features=node_dim,out_features=latent_dim)
        self.multi_head_attention_linear=nn.Linear(in_features=2*latent_dim,out_features=num_head,bias=False)
        self.leaky_relu=nn.LeakyReLU(0.2)
        self.relu=nn.ReLU()
        self.latent_dim=latent_dim
        self.num_head=num_head
        self.is_final_layer=is_final_layer

    def message(self,x_i,x_j,index):
        """
        x_j: source node, [num_edges,node_feat_dim]
        x_i: target node, [num_edges,node_feat_dim]
        index: target node indices, [num_edges,]
        """
        x=torch.cat([x_i,x_j],dim=-1)
        e=self.multi_head_attention_linear(x)
        e=self.leaky_relu(e)

        alpha=scatter_softmax(src=e,index=index,dim=0) # [num_edges,num_head] 
        output=alpha.unsqueeze(-1)*x_j.unsqueeze(1) # [num_edges,num_head,latent_dim]

        return output

    def aggregate(self,inputs,index):
        """
        inputs: [num_edges,num_head,latent_dim]
        index:  target node indices, [num_edges,]
        """
        output=scatter(src=inputs,index=index,dim=0,reduce='sum')  # [num_nodes,num_head,latent_dim]
        if self.is_final_layer:
            output=output.mean(dim=1)  # [num_nodes,latent_dim], logit
        else: 
            output=self.relu(output)
            output=output.view(-1,self.num_head*self.latent_dim)  # [num_nodes,num_head*latent_dim], concat
        return output

    def forward(self,x,edge_index):
        h_0=self.initial_linear(x)
        h=self.propagate(edge_index=edge_index,x=h_0,index=edge_index[1])
        return h

class GAT_classifier(nn.Module):
    def __init__(self,node_dim,latent_dim,num_class,num_head=1,processor:Literal['custom','pyg']='custom'):
        super().__init__()
        if processor=='custom':
            self.processor=Custom_GAT_layer(node_dim=node_dim,latent_dim=latent_dim,num_head=num_head,is_final_layer=True)
        else:
            self.processor=GATConv(in_channels=node_dim,out_channels=latent_dim,heads=num_head,concat=False)
        self.linear=nn.Linear(in_features=latent_dim,out_features=num_class)
    def forward(self,x,edge_index,batch):
        h=self.processor(x,edge_index) # [num_nodes,latent_dim]
        h_graph=global_mean_pool(x=h,batch=batch)  # [num_graphs,latent_dim]
        output=self.linear(h_graph)  # [num_graphs,num_class]
        return output