import torch
import torch.nn as nn
from data import DataManager
from model import NeuralNetwork

import wandb

# start a new wandb run to track this script
wandb.init(
    
    # set the wandb project where this run will be logged
    project="my-toy-project",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "epochs": 10,
    }
    
)

# get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {DEVICE} device")



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            
            wandb.log({
                "loss": loss
            })
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    
    wandb.log({
        "test error": correct,
        "avg loss": test_loss
    })
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
def train_test():
    
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate
    
    data_manager = DataManager(batch_size=64)
    ds_loader_train = data_manager.get_dataloader(train=True)
    ds_loader_test = data_manager.get_dataloader(train=False)
    
    model = NeuralNetwork().to(DEVICE)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        
        print(f"Epoch {t+1}\n-------------------------------")
        
        train(
            dataloader=ds_loader_train,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        
        test(
            dataloader=ds_loader_test,
            model=model,
            loss_fn=loss_fn
        )
        
    print("Done!")


if __name__ == "__main__":
    
    train_test()