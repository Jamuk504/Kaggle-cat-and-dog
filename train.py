import torch
import os
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
from config import train_transform, valid_transform, DEVICE, NUM_WORKERS, LOGGING_INTERVAL, MODEL_TO_TEST, L4_LR, FC_LR
from model import model

def train_one_epoch(epoch_index, tb_writer, model, optimizer, loss_fn, training_loader):
    running_loss = 0.
    last_loss = 0.
    model.train(True)

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % LOGGING_INTERVAL == LOGGING_INTERVAL - 1:
                last_loss = running_loss / 100
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
    if running_loss > 0 or (i % LOGGING_INTERVAL) != (LOGGING_INTERVAL - 1):
        if running_loss > 0:
            last_loss = running_loss / ((i % LOGGING_INTERVAL) + 1)
        elif i == 0:
            last_loss = running_loss / len(training_loader)

    return last_loss

if __name__ == '__main__':
    model_path = MODEL_TO_TEST
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Successfully loaded model state from {model_path}. Resuming training...")
        except Exception as e:
            print(f"Warning: Could not load model state dict from {model_path}. Starting fresh. Error: {e}")
            
    model = model.to(DEVICE)

    training_set = datasets.ImageFolder(root='data/train', transform=train_transform)
    validation_set = datasets.ImageFolder(root='data/valid', transform=valid_transform)

    training_loader = DataLoader(
        training_set,
        batch_size=16,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    validation_loader = DataLoader(
        validation_set,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': L4_LR},
        {'params': model.fc.parameters(), 'lr': FC_LR}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/cat_dog_trainer_{}'.format(timestamp))

    epoch_number = 0
    EPOCHS = 100
    best_vloss = 1_000_000.
    global_best_vloss_file = 'best_vloss.txt'
    if os.path.exists(global_best_vloss_file):
        try:
            with open(global_best_vloss_file, 'r') as f:
                global_best_vloss = float(f.read().strip())
            best_vloss = global_best_vloss
            print(f"Loaded previous global best validation loss: {best_vloss:.4f}")
        except Exception as e:
            print(f"Warning: Could not read global best loss file. Starting fresh. Error: {e}")

    patience = 10
    min_delta = 0.0001
    patience_counter = 0
    current_run_best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        avg_loss = train_one_epoch(
            epoch_number, writer, model, optimizer, loss_fn, training_loader
        )

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(DEVICE)
                vlabels = vlabels.to(DEVICE)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        scheduler.step(avg_vloss)
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < (best_vloss-min_delta):
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            with open(global_best_vloss_file, 'w') as f:
                f.write(str(best_vloss.item()))
            print('Best model so far, saving to {}'.format(model_path))

        if avg_vloss < (current_run_best_vloss - min_delta):
            current_run_best_vloss = avg_vloss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping after {} epochs without improvement'.format(patience))
                break
        epoch_number += 1

