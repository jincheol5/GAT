import torch
import torch.nn as nn
from tqdm import tqdm
from .data_loader import GraphDataLoader

class ModelTrainer:
    @staticmethod
    def train(
            model:nn.Module,
            node_ft:torch.Tensor,
            train_data_loader:GraphDataLoader,
            val_data_loader:GraphDataLoader,
            **args
        ):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
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
        for epoch in tqdm(range(args["epoch"]),desc=f"Training..."):
            model.train()
            for batch_edge_index in tqdm(
                    train_data_loader.get_batch_list(batch_size=args["batch_size"]),
                    desc=f"Training epoch: {epoch}..."
                ):
                """
                """

