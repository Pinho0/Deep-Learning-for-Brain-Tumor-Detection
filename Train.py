"""
train.py

This script trains a CNN to classify brain tumors from MRI scans using PyTorch.
It includes data preparation, model definition, training, evaluation,
checkpoint saving, and early stopping.

"""

# ---------------------
# 1. Library Imports
# ---------------------

from tqdm import tqdm 
import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim

# ---------------------
# 2. Device Setup
# ---------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------
# 3. Dataset Paths and Transforms
# ---------------------

train_dir = r'C:\Users\josep\Desktop\Programar\ML\Brain\dataset\Training'
test_dir = r'C:\Users\josep\Desktop\Programar\ML\Brain\dataset\Testing'

class ConvertImage:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img   
    
transform = transforms.Compose(
    [
        ConvertImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # This parameters were chosen based on the results from EDA.py
        transforms.Normalize(mean=[0.1855, 0.1856, 0.1856], std=[0.2003, 0.2003, 0.2003]), 
    ]
)

# ---------------------
# 4. Dataset and DataLoader Initialization
# ---------------------

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

# ---------------------
# 5. Training Utilities
# ---------------------

epochs_to_train = 20

def train_model(dataloader, device, optimizer, loss_fn, model):   
    model.train() 
    training_loss = 0.0 
    for data, label in tqdm(dataloader): 
        optimizer.zero_grad() 
        data = data.to(device)
        label = label.to(device) 
        
        output = model(data) 
        loss = loss_fn(output, label) 
        loss.backward() 
        optimizer.step() 
        training_loss += loss.item() * data.size(0)
        
    return training_loss / len(dataloader.dataset)

def predict(model, dataloader, device):
    model.eval()
    prob = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, _ in tqdm(dataloader):
            data = data.to(device)
            output = model(data)
            out_prob = nn.functional.softmax(output, dim=1)
            prob = torch.cat((prob, out_prob), dim=0)
    return prob

def loss_accuracy(model, dataloader, loss_fn, device):
    total_loss = 0
    total_correct = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_fn(output, label)
            total_loss += loss.item() * data.size(0)
            correct = torch.eq(torch.argmax(output, dim=1), label)
            total_correct += torch.sum(correct).item()
    n_observations = dataloader.batch_size * len(dataloader)
    accuracy = total_correct / n_observations
    average_loss = total_loss / n_observations
    return average_loss, accuracy


def checkpointing(validation_loss, best_val_loss, model, optimizer, save_path):
    if validation_loss < best_val_loss:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": validation_loss,
        }, save_path)
        

def check_early_stopping(validation_loss, best_val_loss, counter):
    stop = False
    if validation_loss < best_val_loss:
        counter = 0
    else:
        counter += 1
    if counter >= 5:
        stop = True
    return counter, stop

def train( 
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=20,
    device="cpu",
    scheduler=None,
    checkpoint_path=None,
    early_stopping=None,
):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    best_val_loss = float("inf")
    early_stopping_counter = 0

    print("Model evaluation before start of training...")

    train_loss, train_accuracy = loss_accuracy(model, train_loader, loss_fn, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    validation_loss, validation_accuracy = loss_accuracy(model, val_loader, loss_fn, device)
    val_losses.append(validation_loss)
    val_accuracies.append(validation_accuracy)

    for epoch in range(1, epochs + 1):
        print("\n")
        print(f"Starting epoch {epoch}/{epochs}")
        train_model(train_loader, device, optimizer, loss_fn, model)

        train_loss, train_accuracy = loss_accuracy(model, train_loader, loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        validation_loss, validation_accuracy = loss_accuracy(model, val_loader, loss_fn, device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy*100:.4f}%")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy*100:.4f}%")

        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)
        if scheduler:
            scheduler.step()         
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
        if checkpoint_path:
            checkpointing(
                validation_loss, best_val_loss, model, optimizer, checkpoint_path
            )

        if early_stopping:
            early_stopping_counter, stop = early_stopping(
                validation_loss, best_val_loss, early_stopping_counter
            )
            if stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    return (
        learning_rates,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        epoch,
    )

# ---------------------
# 6. Model Architecture
# ---------------------

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(16 * 28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 4)
)

# ---------------------
# 7. Optimizer, Loss Function, and Scheduler
# ---------------------

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------------
# 8. Model Training
# ---------------------

train_results = train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=epochs_to_train,
    device=device,
    scheduler=scheduler,
    checkpoint_path="models/self_model.pth",
    early_stopping=check_early_stopping,
)

# ---------------------
# 9. Training Results Unpacking
# ---------------------

(
    learning_rates_self,
    train_losses_self,
    valid_losses_self,
    train_accuracies_self,
    valid_accuracies_self,
    epochs,
) = train_results