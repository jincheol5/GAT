import os
import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import GAT_classifier
from model_train import ModelTrainer

random.seed(1)
np.random.seed(1)
torch.manual_seed(1) 
os.environ["PYTHONHASHSEED"]=str(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False

config={
    'optimizer':'adam',
    'epochs':100,
    'lr':0.001
}

dataset=TUDataset(root='/tmp/ENZYMES',name='ENZYMES')
num_class=dataset.num_classes
data=dataset[0]
node_dim=data.x.size(1)
latent_dim=32

train_dataset=dataset[:540]
test_dataset=dataset[540:]


data_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)

train_data_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_data_loader=DataLoader(test_dataset,batch_size=16,shuffle=True)

custom_model=GAT_classifier(node_dim=node_dim,latent_dim=latent_dim,num_class=num_class,num_head=3,processor='custom')
pyg_model=GAT_classifier(node_dim=node_dim,latent_dim=latent_dim,num_class=num_class,num_head=3,processor='pyg')

custom_epoch_loss=ModelTrainer.train(model=custom_model,data_loader=train_data_loader,config=config)
custom_acc=ModelTrainer.test(model=custom_model,data_loader=test_data_loader)
pyg_epoch_loss=ModelTrainer.train(model=pyg_model,data_loader=train_data_loader,config=config)
pyg_acc=ModelTrainer.test(model=pyg_model,data_loader=test_data_loader)

print(f"Custom model epoch loss: {custom_epoch_loss}")
print(f"PyG model epoch loss: {pyg_epoch_loss}")
print(f"Custom model Acc: {custom_acc}")
print(f"PyG model Acc: {pyg_acc}")