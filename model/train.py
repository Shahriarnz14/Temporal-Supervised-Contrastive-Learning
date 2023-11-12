import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.contrastive_losses import SupConLoss
from utils.utils import save_checkpoint, load_checkpoint
import datetime
import os
from tqdm import tqdm
import joblib

############################################################
################# Temporal_SCL Training ####################
############################################################

def train_temporal_SCL_main_phase(model_scl, model_clf, model_temporal,
                                  train_loader, valid_loader, num_classes, device='cpu',
                                  num_epochs_scl=100, num_epochs_clf=100, num_epochs_temporal=1,
                                  temperature_scl=0.1,
                                  lr_scl=1e-3, lr_clf=1e-3, lr_temporal=1e-3, 
                                  do_nn_pairing=False, nn_pairing_k=2,
                                  criterion_temporal="MSE", alpha_temporal=100., temporal_l2_reg_coeff=None, predict_raw_temporal=False,
                                  use_nn_clf=False, knn_clf=2,
                                  tensorboard_filename=None, experiment_name='synthetic', verbose=True):
    
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
    model_temporal = model_temporal.to(device)
    if criterion_temporal == "MSE":
        criterion_temporal = nn.MSELoss()
    elif criterion_temporal == "L1":
        criterion_temporal = nn.L1Loss()
    else:
        Warning(f"Criterion {criterion_temporal} is not supported. Using MSE instead.")
        criterion_temporal = nn.MSELoss()
    
    if verbose: 
        print('='*24)
        print("Training the Encoder Model Using Temporal-Supervised-Contrastive Learning")
        print('='*24)
    
    optimizer_temporal_scl = torch.optim.Adam([{"params": model_scl.parameters()}, 
                                      {"params": model_temporal.parameters(), "lr": lr_temporal}
                                     ], lr=lr_scl)
    optimizer_temporal = torch.optim.Adam(model_temporal.parameters(), lr=lr_temporal)
    train_scl_temporal(model_scl=model_scl, model_temporal=model_temporal, 
                       optimizer=optimizer_temporal_scl, optimizer_temporal=optimizer_temporal,
                       train_loader=train_loader, valid_loader=valid_loader, num_classes=num_classes, device=device,
                       num_epochs_scl=num_epochs_scl, num_epochs_temporal=num_epochs_temporal,
                       temperature_scl=temperature_scl, 
                       do_nn_pairing=do_nn_pairing, nn_pairing_k=nn_pairing_k,
                       criterion_temporal=criterion_temporal, alpha_temporal=alpha_temporal, temporal_l2_reg_coeff=temporal_l2_reg_coeff, predict_raw_temporal=predict_raw_temporal, 
                       writer=writer, model_path=model_path, verbose=verbose)
    valid_loss_scl = load_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_scl_temporal.pt", model_scl, optimizer_temporal_scl, device)
    
    if verbose: 
        print('='*24)
        print(f"Completed Training the Encoder Network of Temporal-SCL. (Valid-Loss: {valid_loss_scl})")
        print('='*24)
    

    if verbose: 
        print('+'*24)
        print("Training the Predictor Model Classifier.")
        print('+'*24)
    
    if use_nn_clf:    
        train_nn_clf_temporal(model_scl, model_clf, train_loader, writer, device, verbose)
        model_clf = joblib.load(model_path + f"{writer.get_logdir().split('/')[-1]}_nn_clf_temporal.pkl")
        valid_loss_str = ""
    else:     
        optimizer_clf = torch.optim.Adam(model_clf.parameters(), lr=lr_clf)
        train_clf_temporal(model_clf=model_clf, model_scl=model_scl, optimizer=optimizer_clf, num_epochs=num_epochs_clf, 
                           train_loader=train_loader, valid_loader=valid_loader, device=device, 
                           writer=writer, model_path=model_path, verbose=verbose)
        valid_loss_clf = load_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_clf_temporal.pt", model_clf, optimizer_clf, device)
        valid_loss_str = f"(Valid-Loss: {valid_loss_clf})"
    
    if verbose: 
        print('='*24)
        print(f"Completed Training the Predictor Network. {valid_loss_str}")
        print('='*24)

    writer.close()


def train_scl_temporal(model_scl, model_temporal, 
                       optimizer, optimizer_temporal,
                       train_loader, valid_loader, num_classes, device,
                       num_epochs_scl, num_epochs_temporal,
                       temperature_scl, 
                       do_nn_pairing, nn_pairing_k,
                       criterion_temporal, alpha_temporal, temporal_l2_reg_coeff, predict_raw_temporal, 
                       writer, model_path, verbose,
                       is_ssl=False):
                       
    
    # Check if SSL is enabled and valid training setup
    if is_ssl and (not do_nn_pairing):
        raise ValueError("For SSL, do_nn_pairing must be True as we don't have access to the true labels.")

    
    # Supervised Contrastive Loss
    criterion = SupConLoss(temperature=temperature_scl) 

    ### Train the model ###

    # Burn-in Temporal Network
    if model_temporal is not None:
        model_temporal.train()
        for _ in tqdm(range(num_epochs_temporal)):
            for i, (_, _, _, _, _, _, seq_hist, _) in enumerate(train_loader):
                seq_hist_x = seq_hist[0][0]
                seq_hist_y = seq_hist[0][1]
                seq_hist_len = seq_hist[1]

                if seq_hist_len[-1] == 0:
                    first_empty_tensor_idx = (seq_hist_len == 0).nonzero(as_tuple=True)[0][0].item()
                    seq_hist_x = seq_hist_x[:first_empty_tensor_idx]
                    seq_hist_y = seq_hist_y[:first_empty_tensor_idx]
                    seq_hist_len = seq_hist_len[:first_empty_tensor_idx]

                seq_hist_x = seq_hist_x.to(device)
                seq_hist_y = seq_hist_y.to(device)
                seq_hist_len = seq_hist_len.to(device)

                embedding_temporal_x = model_scl(seq_hist_x)
                embedding_temporal_y = model_scl(seq_hist_y)
                output_temporal_network = model_temporal(embedding_temporal_x.detach(), seq_hist_len)

                # predict next time step embedding vs next raw feature
                if predict_raw_temporal:
                    loss_temporal_only = criterion_temporal(output_temporal_network, seq_hist_y.detach()) # MSELoss of Raw Feature Prediction
                else:
                    loss_temporal_only = criterion_temporal(output_temporal_network, embedding_temporal_y.detach()) # MSELoss
                
                optimizer_temporal.zero_grad()
                loss_temporal_only.backward()
                optimizer_temporal.step()
    # end of Temporal Network Burn-in
    
    # Main Training Loop
    running_loss, running_loss_scl, running_loss_temporal = 0.0, 0.0, 0.0
    eval_every_step_counter = 0
    best_valid_loss = np.inf
    total_step = len(train_loader)

    for epoch in tqdm(range(num_epochs_scl)):
        model_scl.train()
        model_temporal.train()
        for i, (seq, _, seq_idx, seq_label, _, traj_len, seq_hist, seq_pair) in enumerate(train_loader):
            
            _, cnts = seq_label.unique(return_counts=True)
            if do_nn_pairing is False:
                if len(cnts) < (num_classes+1): # some outcomes not in batch
                    if len(cnts) == num_classes:
                        if verbose: print("WARNING: States are assumed to be known for all time-steps. We have no additional hiddent state.")
                    else:
                        raise ValueError('Increase batch-size, some classes are absent in batch')
                elif (cnts==1).any():
                    if verbose: print("Loss needs at least two examples per class")
                    seq, _, seq_idx, seq_label, _, traj_len, seq_pair = duplicate_single_example_in_batch(seq, None, seq_idx, seq_label, None, traj_len, seq_pair, cnts)
            
            # Move tensors to the configured device
            seq = seq.to(device)
            seq_label = seq_label.to(device)
            seq_idx = seq_idx.to(device)
            traj_len = traj_len.to(device)

            # Forward pass
            embedding_outputs = model_scl(seq)
            embedding_outputs = torch.unsqueeze(embedding_outputs, dim=1)

            # Either use SCL loss with the true labels in seq_label or SimCLR loss with NN pairings in seq_pair
            if do_nn_pairing is False:
                loss_supervised_contrast = criterion(embedding_outputs, seq_label)
            else: # Nearest Neighbors Mask
                loss_supervised_contrast = criterion(embedding_outputs, seq_pair) 
            
            running_loss_scl += loss_supervised_contrast.item()
            loss = loss_supervised_contrast 
            
            # Regularize the Embedding Space using Temporal Network
            if model_temporal is not None:
                seq_hist_x = seq_hist[0][0]
                seq_hist_y = seq_hist[0][1]
                seq_hist_len = seq_hist[1]

                # Ignore short-length sequences that result in empty tensors
                if seq_hist_len[-1] == 0:
                    first_empty_tensor_idx = (seq_hist_len == 0).nonzero(as_tuple=True)[0][0].item()
                    seq_hist_x = seq_hist_x[:first_empty_tensor_idx]
                    seq_hist_y = seq_hist_y[:first_empty_tensor_idx]
                    seq_hist_len = seq_hist_len[:first_empty_tensor_idx]

                seq_hist_x = seq_hist_x.to(device)
                seq_hist_y = seq_hist_y.to(device)
                seq_hist_len = seq_hist_len.to(device)

                embedding_temporal_x = model_scl(seq_hist_x)
                embedding_temporal_y = model_scl(seq_hist_y)
                output_temporal_network = model_temporal(embedding_temporal_x, seq_hist_len)

                # predict next time step embedding vs next raw feature
                if predict_raw_temporal:
                    loss_temporal = criterion_temporal(output_temporal_network, seq_hist_y)
                else:
                    loss_temporal = criterion_temporal(output_temporal_network, embedding_temporal_y)
                
                running_loss_temporal += loss_temporal.item()
                loss += alpha_temporal * loss_temporal

                # L2 Regularization on Temporal Network Weights
                if temporal_l2_reg_coeff is not None:
                    loss += temporal_l2_reg_coeff * sum(p.pow(2.0).sum() for p in model_temporal.parameters())
            # end of Temporal Network Regularization


            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            eval_every_step_counter  += 1
            running_loss += loss.item()

            # Tensorboard Logging
            if ((i+1) % 100 == 0) or ((i == 0) and (epoch == 0)) or ((i+1) == len(train_loader)):
                if verbose:
                    print('Epoch [{}/{}], Step [{}/{}], Training-Loss: {:.4f}, SCL: {:.4f}, Temporal: {:.4f}'
                          .format(epoch+1, num_epochs_scl, i+1, total_step, loss.item(), loss_supervised_contrast.item(), loss_temporal.item()))
                writer.add_scalar("Train-TSCL/Encoder - Training Loss", running_loss/eval_every_step_counter, epoch*total_step+i)
                writer.add_scalar("Train-TSCL/Encoder - Training SCL Loss", running_loss_scl/eval_every_step_counter, epoch*total_step+i)
                if model_temporal is not None: writer.add_scalar("Train-TSCL/Encoder - Training Temporal Loss", running_loss_temporal/eval_every_step_counter, epoch*total_step+i)
                running_loss, running_loss_scl, running_loss_temporal = 0.0, 0.0, 0.0
                eval_every_step_counter = 0
                
        # Validation Set Evaluation
        if ((epoch+1) % 10 == 0) or (epoch == 0) or ((epoch+1) == num_epochs_scl):
            
            valid_running_loss, valid_running_loss_scl = run_validation(model_scl, model_temporal, valid_loader, num_classes, 
                                                                        criterion, device, verbose)
                    
            writer.add_scalar("Train-TSCL/Encoder - Validation Loss", valid_running_loss/len(valid_loader), epoch)
            writer.add_scalar("Train-TSCL/Encoder - Validation SCL Loss (Label)", valid_running_loss_scl/len(valid_loader), epoch)
                
            average_valid_loss = valid_running_loss/len(valid_loader)
            if average_valid_loss < best_valid_loss:
                best_valid_loss = average_valid_loss
                save_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_scl_temporal.pt", model_scl, optimizer, best_valid_loss)


def run_validation(model_scl, model_temporal, valid_loader, num_classes, criterion, device, verbose):
    valid_running_loss, valid_running_loss_scl, valid_running_loss_temporal = 0.0, 0.0, 0.0
    with torch.no_grad():                    
        # validation loop
        model_scl.eval()
        if model_temporal is not None:
            model_temporal.eval()
        for i, (seq, _, _, seq_label, _, traj_len) in enumerate(valid_loader):
            _, cnts = seq_label.unique(return_counts=True)
            if len(cnts) < (num_classes+1): # some outcomes not in batch
                if len(cnts) == num_classes:
                    if verbose: print("WARNING: States are assumed to be known for all time-steps. We have no additional hiddent state.")
                else:
                    raise ValueError('Increase batch-size (Validation), some classes are absent in batch')
            elif (cnts==1).any():
                seq, _, _, seq_label, _, traj_len, _ = duplicate_single_example_in_batch(seq, None, None, seq_label, None, traj_len, None, cnts)

            seq = seq.to(device)
            seq_label = seq_label.to(device)
            traj_len = traj_len.to(device)

            # Forward pass
            embedding_outputs = model_scl(seq)
            embedding_outputs = torch.unsqueeze(embedding_outputs, dim=1)

            loss_supervised_contrast = criterion(embedding_outputs, seq_label)
            valid_running_loss_scl += loss_supervised_contrast.item()

            loss = loss_supervised_contrast                  
            valid_running_loss += loss.item()
    
    return valid_running_loss, valid_running_loss_scl


def train_clf_temporal(model_clf, model_scl, optimizer, num_epochs, 
                       train_loader, valid_loader, device, 
                       writer, model_path, verbose=True):
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    model_clf = model_clf.to(device)
    
    model_scl.eval()
    with torch.no_grad():
        train_data = []
        for i, (seq, _, _, _, seq_outcome, _, _, _) in enumerate(train_loader):
            outcome_sequence = seq.to(device)
            outcome_label = seq_outcome.to(device)
            embedding_outputs = model_scl(outcome_sequence)
            train_data.append((embedding_outputs, outcome_label))
            
        valid_data = []
        for i, (seq, _, _, _, seq_outcome, _) in enumerate(valid_loader):
            outcome_sequence = seq.to(device)
            outcome_label = seq_outcome.to(device)
            embedding_outputs = model_scl(outcome_sequence)
            valid_data.append((embedding_outputs, outcome_label))
    
    # Train the model
    running_loss = 0.0
    eval_every_step_counter = 0
    best_valid_loss = np.inf
    total_step = len(train_loader)
    for epoch in tqdm(range(num_epochs)):
        for i, (embedding_outputs, outcome_label) in enumerate(train_data):
            model_clf.train()
            
            # Forward pass
            outputs = model_clf(embedding_outputs)
            loss = criterion(outputs, outcome_label)
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            if (epoch == (num_epochs-1)) and (i == (len(train_data)-1)):
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            optimizer.step()
            
            eval_every_step_counter  += 1
            running_loss += loss.item()
            
            # Tensorboard Logging
            if ((i+1) % 100 == 0) or ((i == 0) and (epoch == 0)) or ((i+1) == len(train_loader)):
                if verbose:
                    print ('Epoch [{}/{}], Step [{}/{}], Training-Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                writer.add_scalar("Train-TSCL/Predictor - Training Loss CLF", running_loss/eval_every_step_counter, epoch*total_step+i)
                running_loss = 0.0
                eval_every_step_counter = 0
                
                # Validation Set Evaluation
                valid_running_loss = 0.0
                with torch.no_grad():                    
                    # validation loop
                    model_clf.eval()
                    for i, (embedding_outputs, outcome_label) in enumerate(valid_data):
                        # Forward pass
                        outputs = model_clf(embedding_outputs)
                        loss = criterion(outputs, outcome_label)
                        valid_running_loss += loss.item()

                    writer.add_scalar("Train-TSCL/Predictor - Validation Loss CLF", valid_running_loss/len(valid_loader), epoch*total_step+i)

                    average_valid_loss = valid_running_loss/len(valid_loader)
                    if average_valid_loss < best_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(model_path + f"{writer.get_logdir().split('/')[-1]}_clf_temporal.pt", model_clf, optimizer, best_valid_loss)


def train_nn_clf_temporal(model_scl, model_clf, train_loader, device, writer, model_path, verbose):
    train_data_Z, train_data_Y = [], []
    with torch.no_grad():
        model_scl.eval()
        for i, (seq, _, _, _, seq_outcome, _, _, _) in enumerate(train_loader):
            outcome_sequence = seq.to(device)
            outcome_label = seq_outcome.to(device)
            embedding_outputs = model_scl(outcome_sequence)
            train_data_Z.append(embedding_outputs.cpu().numpy())
            train_data_Y.append(outcome_label.cpu().numpy())

    training_data_Z = np.concatenate(train_data_Z)
    training_data_Y = np.concatenate(train_data_Y)

    # model_clf = NearestNeighbors(n_neighbors=knn_clf, n_jobs=-1)
    model_clf.fit(training_data_Z, training_data_Y)
    if verbose: print("Saving Classic-CLF ...")
    joblib.dump(model_clf, model_path + f"{writer.get_logdir().split('/')[-1]}_nn_clf_temporal.pkl")
    if verbose: print("Saving Classic-CLF Completed!")


def get_loss_temporal_spacing(embedding_outputs, centroids_, seq_idx, traj_len, tau):
    return torch.mean( (1-(embedding_outputs*centroids_).sum(dim=1)) * (tau**(traj_len-seq_idx-1)) )


def duplicate_single_example_in_batch(seq, seq_last_timestep, seq_idx, seq_label, seq_outcome, traj_len, seq_pair, cnts_per_class):
    single_class_examples = np.where(cnts_per_class==1)[0]
    for single_class_ in single_class_examples:
        idx_single_class_example = np.where(seq_label == single_class_)[0][0]
        seq = torch.cat((seq, seq[idx_single_class_example].unsqueeze(dim=0)), dim=0) if seq is not None else None
        seq_last_timestep = torch.cat((seq_last_timestep, seq_last_timestep[idx_single_class_example].unsqueeze(dim=0)), dim=0) if seq_last_timestep is not None else None
        seq_idx = torch.cat((seq_idx, seq_idx[idx_single_class_example].unsqueeze(dim=0)), dim=0) if seq_idx is not None else None
        seq_label = torch.cat((seq_label, seq_label[idx_single_class_example].unsqueeze(dim=0)), dim=0) if seq_label is not None else None
        seq_outcome = torch.cat((seq_outcome, seq_outcome[idx_single_class_example].unsqueeze(dim=0)), dim=0) if seq_outcome is not None else None
        traj_len = torch.cat((traj_len, traj_len[idx_single_class_example].unsqueeze(dim=0)), dim=0) if traj_len is not None else None
        seq_pair = torch.cat((seq_pair, seq_pair[idx_single_class_example].unsqueeze(dim=0)), dim=0) if seq_pair is not None else None
    return seq, seq_last_timestep, seq_idx, seq_label, seq_outcome, traj_len, seq_pair