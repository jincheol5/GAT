import torch
import networkx as nx

class Graph:
    def __init__(self,
            nx_graph:nx.DiGraph,
            node_dim:int=32
        ):
        """
        """
        self.nx_graph=nx_graph
        # remove self-loop
        nx_graph.remove_edges_from(
            nx.selfloop_edges(nx_graph)
        )

        node_mapping={
            node_id:idx
            for idx,node_id in enumerate(sorted(nx_graph.nodes()))
        }
        nx_graph=nx.relabel_nodes(nx_graph,node_mapping)

        self.node_ft=torch.ones(
            len(node_mapping),
            node_dim
        ) # [N,node_dim]
        
        self.edge_index=torch.tensor(
            [
                [node_mapping[src],node_mapping[tar]]
                for src,tar in nx_graph.edges()
            ],
            dtype=torch.long
        ).t().contiguous() # [2,E]

    def check_mapped(self):
        num_nodes=self.nx_graph.number_of_nodes()
        is_mapped=set(self.nx_graph.nodes())==set(range(num_nodes))
        return is_mapped

    def set_graph(self,
            graph:nx.DiGraph,
            node_dim:int=32
        ):
        """
        """
        # remove self-loop
        graph.remove_edges_from(
            nx.selfloop_edges(graph)
        )

        node_mapping={
            node_id:idx
            for idx,node_id in enumerate(sorted(graph.nodes()))
        }
        graph=nx.relabel_nodes(graph,node_mapping)

        self.node_ft=torch.ones(
            len(node_mapping),
            node_dim
        ) # [N,node_dim]
        
        self.edge_index=torch.tensor(
            [
                [node_mapping[src],node_mapping[tar]]
                for src,tar in graph.edges()
            ],
            dtype=torch.long
        ).t().contiguous() # [2,E]

    def get_node_ft(self,):
        return self.node_ft # [N,node_dim]
    
    def get_edge_index(self,):
        return self.edge_index # [2,E]


