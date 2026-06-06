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
            num_node=5
            edge_index=torch.tensor(
                [
                    [0,0,1,1,2,2,3,4],
                    [0,1,1,2,2,0,3,4]
                ],
                dtype=torch.long
            )
            idx,population=SamplingUtils.convert_edge_index_to_idx(
                num_node=num_node,
                edge_index=edge_index
            )
            print(f"num_node: {num_node}",end="\n\n")
            print(f"edge_index: {edge_index}",end="\n\n")
            print(f"idx: {idx}",end="\n\n")
            print(f"population: {population}")
        
        case 2:
            """
            Test. SamplingUtils.negative_sampling
            """
            num_node=5
            edge_index=torch.tensor(
                [
                    [0,0,1,1,2,2,3,4],
                    [0,1,1,2,2,0,3,4]
                ],
                dtype=torch.long
            )
            neg_edge_index=SamplingUtils.negative_sampling(
                num_node=num_node,
                edge_index=edge_index,
                num_neg_edge=4
            )
            print(f"num_node: {num_node}",end="\n\n")
            print(f"all edge_index: ")
            print(edge_index,end="\n\n")
            print(f"neg_edge_index: ")
            print(neg_edge_index,end="\n\n")

        case 3:
            """
            Test. SamplingUtils.neighbor_sampling
            """
            edge_index=torch.tensor(
                [
                    [0,0,0,1,1,2,2,3,3,4,4],  # src
                    [0,1,2,1,3,2,3,3,4,4,5]   # dst
                ],
                dtype=torch.long
            )
            pos_edge_index=torch.tensor(
                [
                    [4],
                    [5]
                ],
                dtype=torch.long
            )
            neg_edge_index=torch.tensor(
                [
                    [1],
                    [5]
                ],
                dtype=torch.long
            )
            sub_edge_index=SamplingUtils.neighbor_sampling(
                    edge_index=edge_index,
                    pos_edge_index=pos_edge_index,
                    neg_edge_index=neg_edge_index,
                    n_hop=1
                )
            print(f"1-hop sub_edge_index: ")
            print(sub_edge_index,end="\n\n")

            sub_edge_index=SamplingUtils.neighbor_sampling(
                    edge_index=edge_index,
                    pos_edge_index=pos_edge_index,
                    neg_edge_index=neg_edge_index,
                    n_hop=2
                )
            print(f"2-hop sub_edge_index: ")
            print(sub_edge_index,end="\n\n")
            """
            답안:
            1-hop sub_edge_index:
            tensor([[0, 1, 3, 4, 4],
                    [1, 1, 4, 4, 5]])

            2-hop sub_edge_index:
            tensor([[0, 0, 1, 1, 2, 3, 3, 4, 4],
                    [0, 1, 1, 3, 3, 3, 4, 4, 5]])
            """

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