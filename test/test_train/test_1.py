import argparse
import torch
from utils import DataUtils 
from train import Graph,GraphDataLoader,ModelTrainer
from model import GAT_Link_Prediction

"""
<< Test >> 
train.model_train.ModelTrainer
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. train.model_train.ModelTrainer.train_link_prediction
            """
            # 임시로 train/val/test 분할 없이 test
            nx_graph=DataUtils.convert_dataset_to_nx_graph(dataset_name=f"email-Eu-core")
            graph=Graph(nx_graph=nx_graph,node_dim=4)
            node_ft=graph.get_node_ft()
            edge_index=graph.get_edge_index()
            data_loader=GraphDataLoader(edge_index=edge_index)
            model=GAT_Link_Prediction(
                node_dim=4,
                latent_dim=32,
                output_dim=4
            )
            config={
                "epoch":1,
                "batch_size":100,
                "optimizer":"adam",
                "lr":0.0005
            }
            ModelTrainer.train_link_prediction(
                model=model,
                node_ft=node_ft,
                edge_index=edge_index,
                exclude_edge_index=edge_index,
                train_data_loader=data_loader,
                **config
            )
            print(f"Model train finish!")

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