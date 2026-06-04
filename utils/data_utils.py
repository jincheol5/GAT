import os
import networkx as nx
import torch

class DataUtils:
    base_path=os.path.join("..","data","gat")
    @staticmethod
    def convert_dataset_to_nx_graph(dataset_name:str):
        """
        Input:
            dataset_name: str
        Output:
            graph: nx.DiGraph
        """
        dataset_path=os.path.join("dataset",dataset_name,f"{dataset_name}.txt")
        graph=nx.read_edgelist(
            dataset_path,
            nodetype=int,
            create_using=nx.DiGraph()
        )
        return graph
