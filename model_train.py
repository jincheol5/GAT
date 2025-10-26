import torch
import torch.nn.functional as F
from tqdm import tqdm

class ModelTrainer:
    @staticmethod
    def train(model,data_loader,config:dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        model.train()
        total_epoch_loss=[]
        for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
            epoch_loss=0
            for data in data_loader:
                data=data.to(device)
                output=model(data.x,data.edge_index,data.batch) # [num_graphs,num_class]
                loss=F.cross_entropy(output,data.y)
                epoch_loss+=loss.item()

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_epoch_loss.append(epoch_loss)
        return total_epoch_loss

    @staticmethod
    def test(model,data_loader):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        correct=0
        total=0
        with torch.no_grad():
            for data in tqdm(data_loader,desc=f"Testing..."):
                data=data.to(device)
                output=model(data.x,data.edge_index,data.batch) # [num_graphs,num_class]
                prob=F.softmax(output,dim=1)
                pred=prob.argmax(dim=1) # [num_graphs,]

                print(f"pred: {pred}")
                print()
                print(f"label: {data.y}")

                correct+=int((pred==data.y).sum())
                total+=data.y.size(0)
        return correct/total