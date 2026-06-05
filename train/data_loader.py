import torch

class GraphDataLoader:
    def __init__(self,edge_index:torch.Tensor):
        self.edge_index=edge_index
    
    def get_batch_list(self,batch_size:int=1):
        """
        Input:
            batch_size: int
        Return:
            List of batch_edge_index [2,batch_E]
        """
        num_edges=self.edge_index.size(1)
        batch_list=[]
        for start in range(0,num_edges,batch_size):
            end=min(start+batch_size,num_edges)
            batch_edge_index=self.edge_index[:,start:end]
            batch_list.append(batch_edge_index)
        return batch_edge_index