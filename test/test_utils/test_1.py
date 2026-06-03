import argparse
import torch
from utils import SamplingUtils

"""
<< Test >> 
utils.sampling_utils.SamplingUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. SamplingUtils.convert_edge_index_to_idx
            """
            num_nodes=5
            edge_index=torch.tensor(
                [
                    [0,0,1,1,2,2,3,4],
                    [0,1,1,2,2,0,3,4]
                ],
                dtype=torch.long
            )
            idx,population=SamplingUtils.convert_edge_index_to_idx(
                num_nodes=num_nodes,
                edge_index=edge_index
            )
            print(f"num_nodes: {num_nodes}",end="\n\n")
            print(f"edge_index: {edge_index}",end="\n\n")
            print(f"idx: {idx}",end="\n\n")
            print(f"population: {population}")
        
        case 2:
            """
            Test. SamplingUtils.negative_sampling
            """
            num_nodes=5
            edge_index=torch.tensor(
                [
                    [0,0,1,1,2,2,3,4],
                    [0,1,1,2,2,0,3,4]
                ],
                dtype=torch.long
            )
            neg_edge_index=SamplingUtils.negative_sampling(
                num_nodes=num_nodes,
                edge_index=edge_index
            )
            print(f"num_nodes: {num_nodes}",end="\n\n")
            print(f"edge_index: {edge_index}",end="\n\n")
            print(f"neg_edge_index: {neg_edge_index}",end="\n\n")

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