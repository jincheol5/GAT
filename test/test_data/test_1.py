import argparse
import torch
from modules import Graph
from utils import DataUtils

"""
<< Test >> 
modules.graph
utils.data_utils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. 
            """
            nx_graph=DataUtils.convert_dataset_to_nx_graph(dataset_name=f"email-Eu-core")
            graph=Graph(nx_graph=nx_graph,node_dim=4)
            node_ft=graph.get_node_ft()
            edge_index=graph.get_edge_index()
            print(f"check mapped: {graph.check_mapped()}")
            print(f"node 개수: {node_ft.size(0)}")
            print(f"edge 개수: {edge_index.size(1)}")
            print(f"edge 개수 (self-loop 제외): {edge_index.size(1)-node_ft.size(0)}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)