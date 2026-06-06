import torch
import torch.nn as nn
from tqdm import tqdm
from .graph import Graph
from .data_loader import GraphDataLoader
from utils import SamplingUtils

class ModelTrainer:
    @staticmethod
    def train_link_prediction(
            model:nn.Module,
            node_ft:torch.Tensor,
            edge_index:torch.Tensor,
            exclude_edge_index:torch.Tensor,
            train_data_loader:GraphDataLoader,
            val_data_loader:GraphDataLoader,
            **args
        ):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        exclude_edge_index=exclude_edge_index.to(device=device)
        if args["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=args["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=args["lr"]
            )

        """
        model train
        """
        num_node=node_ft.size(0)
        for epoch in tqdm(range(args["epoch"]),desc=f"Training..."):
            model.train()
            for batch_edge_index in tqdm(
                    train_data_loader.get_batch_list(batch_size=args["batch_size"]),
                    desc=f"Training epoch: {epoch}..."
                ):
                batch_edge_index=batch_edge_index.to(device=device)
                batch_edge_size=batch_edge_index.size(1)
                batch_neg_edge_index=SamplingUtils.negative_sampling(
                    num_node=num_node,
                    edge_index=exclude_edge_index,
                    num_neg_edge=batch_edge_size
                )

                model()

