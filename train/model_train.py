import torch
import torch.nn as nn
from tqdm import tqdm
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
            val_data_loader:GraphDataLoader=None,
            **kwargs
        ):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        node_ft=node_ft.to(device=device)
        edge_index=edge_index.to(device=device)
        exclude_edge_index=exclude_edge_index.to(device=device)
        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        model train
        """
        num_node=node_ft.size(0)
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Training..."):
            model.train()
            batch_count=0
            for batch_edge_index in tqdm(
                    train_data_loader.get_batch_list(batch_size=kwargs["batch_size"]),
                    desc=f"Training epoch: {epoch}..."
                ):
                batch_count+=1

                batch_edge_index=batch_edge_index.to(device=device)
                batch_edge_size=batch_edge_index.size(1)
                batch_neg_edge_index=SamplingUtils.negative_sampling(
                    num_node=num_node,
                    edge_index=exclude_edge_index,
                    num_neg_edge=batch_edge_size
                )
                sub_edge_index=SamplingUtils.neighbor_sampling(
                    num_node=num_node,
                    edge_index=edge_index,
                    pos_edge_index=batch_edge_index,
                    neg_edge_index=batch_neg_edge_index,
                    n_hop=2
                )
                target_edge_index=torch.cat(
                    [batch_edge_index,batch_neg_edge_index],
                    dim=-1
                )
                pos_edge_label=SamplingUtils.get_edge_label(
                    edge_index=batch_edge_index,
                    label=1
                )
                neg_edge_label=SamplingUtils.get_edge_label(
                    edge_index=batch_neg_edge_index,
                    label=0
                )
                target_edge_label=torch.cat(
                    [pos_edge_label,neg_edge_label],
                    dim=0
                ) # [target_E,]

                """
                predict
                """
                pred_target_edge_logit=model(
                    node_ft=node_ft,
                    embed_edge_index=sub_edge_index,
                    target_edge_index=target_edge_index
                ) # [target_E,1]
                pred_target_edge_logit=pred_target_edge_logit.squeeze(-1) # [target_E,]

                """
                Loss
                """
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_target_edge_logit,target_edge_label)

                print(f"{epoch} epoch {batch_count} batch_count loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
