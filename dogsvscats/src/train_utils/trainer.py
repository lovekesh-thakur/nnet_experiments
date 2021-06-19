import torch
import torch.nn as nn
import numpy as np
import torchmetrics
import wandb

def train(model, train_dataloader, optimizer, 
         loss_fn, device = "cpu"):
    """
    Train model.

    Args:
    -------
    model : pytorch model 
    train_dataloader : dataloader for training dataset
    num_epochs : number of epochs to train
    optimizer : optimizer for training
    loss_fn : loss function
    """
    
    loss = []
    iterations = 0
    for X, Y in train_dataloader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        out = model(X)
            
        delta = loss_fn(out, Y)
        delta.backward()
        optimizer.step()

        loss.append(delta.item())
        iterations += 1
        
    return { 'loss' : np.mean(loss), 
              'iterations' : iterations }


def valid(model, data, loss_fn, device = "cpu"):
    """
    Get validation score for validation set
    """
    losses = []
    acc = []
    with torch.no_grad():
        for X, Y in data:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            preds = torch.round(torch.sigmoid(out))
            loss = loss_fn(out, Y)
            losses.append(loss.item())
            res = torch.sum(preds == Y).item()
            tot = Y.shape[0]
            acc.append(res/tot)
    return {"val_loss" : np.mean(losses),  'val_accuracy' :  np.mean(acc)}
            
