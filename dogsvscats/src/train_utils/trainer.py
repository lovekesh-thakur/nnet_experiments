import torch
import numpy as np
import torchmetrics

def train(model, train_dataloader, num_epochs, optimizer, 
         loss_fn, device = "cpu"):
    """
    Train model.

    Args:
    -------
    model : pytorch model 
    train_dataloader : dataloader for training dataset
    num_epochs : number of epochs to train
    optimizer : optimizer for training
    valid_dataloader : validation dataloader 
    """
    iterations = 0
    for epoch in range(num_epochs):
        loss = []
        for X, Y in train_dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            
            delta = loss_fn(out, Y)
            delta.backward()
            optimizer.step()

            loss.append(delta.item())
            if iterations % 101 == 0:
                print(f"At {epoch} epoch iterations {iterations} loss is {np.mean(loss)}")
            iterations += 1


def valid(model, data, loss_fn, device = "cpu"):
    """
    Get validation score for validation set
    """
    losses = []
    metrics = torchmetrics.Accuracy()
    with torch.no_grad():
        for X, Y in data:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            preds = torch.argmax(out)
            loss = loss_fn(out, Y)
            losses.append(loss.items())
            acc = metrics(preds, Y)
    print(f"Loss is {np.mean(losses)} and Accuracy is {metrics.compute()}")
            
