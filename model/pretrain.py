import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.contrastive_losses import SupConLoss
from utils.utils import save_checkpoint, load_checkpoint
import datetime
import os
from tqdm import tqdm


############################################################
################## Temporal-SCL Pretraining ################
############################################################

def train_temporal_scl_pretrain_phase(model_scl, model_clf, 
                                      train_loader, valid_loader, 
                                      num_epochs_scl=100, num_epochs_clf=100, 
                                      temperature_scl=0.1, 
                                      lr_scl=1e-3, lr_clf=1e-3, device='cpu',  
                                      tensorboard_filename=None, experiment_name='synthetic', verbose=True):
    '''
    Pretraining the encoder network and also pretraining the predictor network on top of the pretrained-embeddings (for intermediate evaluation only).
    '''
    
    # Tensorboard: Make unique filename for each run if no name is provided
    if tensorboard_filename is None:
        tensorboard_filename = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # check if results folders and the corresponding paths exist
    results_path = f"../results/{experiment_name}/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    tensorboard_path = f"{results_path}tensorboard/"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    model_path = f"{results_path}saved_models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    writer = SummaryWriter(tensorboard_path + tensorboard_filename)
    
    # Pretrain the Encoder Network
    if num_epochs_scl > 0:
        if verbose: 
            print('='*24)
            print("Pre-training the Encoder Model Using Supervised-Contrastive Learning")
            print('='*24)
        
        optimizer_scl = torch.optim.Adam(model_scl.parameters(), lr=lr_scl)
        pretrain_encoder(model_scl=model_scl, optimizer=optimizer_scl, num_epochs=num_epochs_scl, 
                         train_loader=train_loader, valid_loader=valid_loader, device=device,
                         temperature_scl=temperature_scl, 
                         writer=writer, model_path=model_path, verbose=verbose)
        valid_loss_scl = load_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_scl.pt", model_scl, optimizer_scl, device)
        
        if verbose: 
            print('='*24)
            print(f"Completed Pre-training the Encoder Network. (Valid-Loss: {valid_loss_scl})")
            print('='*24)

    
    # Pretrain the Predictor Network (Note this is for intermediate-evaluation only)
    if num_epochs_clf > 0:
        if verbose: 
            print('+'*24)
            print("Training the Predictor Network Classifier.")
            print('+'*24)
     
        optimizer_clf = torch.optim.Adam(model_clf.parameters(), lr=lr_clf)
        pretrain_predictor(model_clf=model_clf, model_scl=model_scl, optimizer=optimizer_clf, num_epochs=num_epochs_clf, 
                           train_loader=train_loader, valid_loader=valid_loader, device=device, 
                           writer=writer, model_path=model_path, verbose=verbose)
        valid_loss_clf = load_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_clf.pt", model_clf, optimizer_clf, device)
        
        if verbose: 
            print('='*24)
            print(f"Completed Training the (Pre-training) Predictor Network. (Valid-Loss: {valid_loss_clf})")
            print('='*24)
    
    writer.close()
    
    
def pretrain_predictor(model_clf, model_scl, optimizer, num_epochs, 
                 train_loader, valid_loader, device,
                 writer, model_path, verbose=True):
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    model_clf = model_clf.to(device)
    
    # Train the model
    running_loss = 0.0
    eval_every_step_counter = 0
    best_valid_loss = np.inf
    total_step = len(train_loader)

    for epoch in tqdm(range(num_epochs)):
        model_scl.eval()
        for i, (outcome_sequence, outcome_label) in enumerate(train_loader): 
            model_clf.train()
            # Move tensors to the configured device
            outcome_sequence = outcome_sequence.to(device)
            outcome_label = outcome_label.to(device)
            
            # Forward pass
            with torch.no_grad():
                embedding_outputs = model_scl(outcome_sequence)
            outputs = model_clf(embedding_outputs)
            loss = criterion(outputs, outcome_label)
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            eval_every_step_counter  += 1
            running_loss += loss.item()
            
            # Saving and Evaluation of the model on the validation set
            if ((i+1) % 100 == 0) or ((i == 0) and (epoch == 0)) or ((i+1) == len(train_loader)):
                if verbose:
                    print ('Epoch [{}/{}], Step [{}/{}], Training-Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                writer.add_scalar("Loss-SCL/Training Loss - CLF", running_loss/eval_every_step_counter, epoch*total_step+i)
                running_loss = 0.0
                eval_every_step_counter = 0
                
                # Validation Set Evaluation
                valid_running_loss = 0.0
                with torch.no_grad():                    
                    # validation loop
                    model_clf.eval()
                    for i, (outcome_sequence, outcome_label) in enumerate(valid_loader): 
                        outcome_sequence = outcome_sequence.to(device)
                        outcome_label = outcome_label.to(device)

                        # Forward pass
                        embedding_outputs = model_scl(outcome_sequence)
                        outputs = model_clf(embedding_outputs)
                        loss = criterion(outputs, outcome_label)
                        valid_running_loss += loss.item()

                    writer.add_scalar("Loss-SCL/Validation Loss - CLF", valid_running_loss/len(valid_loader), epoch*total_step+i)

                    average_valid_loss = valid_running_loss/len(valid_loader)
                    if average_valid_loss < best_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_clf.pt", model_clf, optimizer, best_valid_loss)


def pretrain_encoder(model_scl, optimizer, num_epochs, 
                     train_loader, valid_loader, device,
                     temperature_scl, 
                     writer, model_path, verbose=True):
    
    # Supervised Contrastive Loss
    criterion = SupConLoss(temperature=temperature_scl)

    # Train the model
    running_loss = 0.0
    eval_every_step_counter = 0
    best_valid_loss = np.inf
    total_step = len(train_loader)

    for epoch in tqdm(range(num_epochs)):
        model_scl.train()
        for i, (outcome_sequence, outcome_label) in enumerate(train_loader):              
            _, cnts = outcome_label.unique(return_counts=True)
            if (cnts<=1).any(): 
                if verbose: print("SCL-Loss needs at least two examples per class to be present in the batch. (Solution: Increase Batch-Size)")
                continue
            
            # Move tensors to the configured device
            outcome_sequence = outcome_sequence.to(device)
            outcome_label = outcome_label.to(device)

            # Forward pass
            embedding_outputs = model_scl(outcome_sequence)
            embedding_outputs = torch.unsqueeze(embedding_outputs, dim=1)

            loss = criterion(embedding_outputs, outcome_label)

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            eval_every_step_counter  += 1
            running_loss += loss.item()

            # Tensorboard Logging
            if ((i+1) % 100 == 0) or ((i == 0) and (epoch == 0)) or ((i+1) == len(train_loader)):
                if verbose:
                    print ('Epoch [{}/{}], Step [{}/{}], Training-Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                writer.add_scalar("PreTrain/Training Loss", running_loss/eval_every_step_counter, epoch*total_step+i)
                running_loss = 0.0
                eval_every_step_counter = 0
                
        # Validation Set Evaluation
        if ((epoch+1) % 10 == 0) or (epoch == 0) or ((epoch+1) == num_epochs):
            valid_running_loss = 0.0
            with torch.no_grad():                    
                # validation loop
                model_scl.eval()
                for i, (outcome_sequence, outcome_label) in enumerate(valid_loader): 
                    _, cnts = outcome_label.unique(return_counts=True)
                    if (cnts<=1).any(): continue
                    
                    outcome_sequence = outcome_sequence.to(device)
                    outcome_label = outcome_label.to(device)

                    # Forward pass
                    embedding_outputs = model_scl(outcome_sequence)
                    embedding_outputs = torch.unsqueeze(embedding_outputs, dim=1)
                    loss = criterion(embedding_outputs, outcome_label)

                    valid_running_loss += loss.item()
                    
                writer.add_scalar("Loss-SCL/Validation Loss", valid_running_loss/len(valid_loader), epoch)
                
                average_valid_loss = valid_running_loss/len(valid_loader)
                if average_valid_loss < best_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_scl.pt", model_scl, optimizer, best_valid_loss)