from utils.evaluation_utils import get_evaluation_metrics
import torch
from torch import nn
import numpy as np   

def evaluate_training(model_scl, model_clf, data_loader, num_classes, device, return_embedding=False, verbose=True):
    # Evaluate Model on Test-Set
    with torch.no_grad():                    
        model_scl.eval()
        model_clf.eval()

        model_clf.to(device)
        model_scl.to(device)
        
        outcome_sequence_list, embedding_outputs_list, outcome_label_lists = np.array([]), np.array([]), np.zeros(0, dtype=int)
        y_preds, y_trues = torch.empty((0, num_classes)), torch.empty((0))
        for i, (outcome_sequence, outcome_label) in enumerate(data_loader):  
            # Move tensors to the configured device
            outcome_sequence = outcome_sequence.to(device)
            outcome_label = outcome_label.to(device)

            # Forward pass
            embedding_outputs = model_scl(outcome_sequence)
            outputs = model_clf(embedding_outputs)

            y_pred = outputs.to('cpu')
            sm_layer = nn.Softmax(dim=1)
            y_pred = sm_layer(y_pred)

            y_preds = torch.cat((y_preds, y_pred.clone().detach().to('cpu')))
            y_trues = torch.cat((y_trues, outcome_label.clone().detach().to('cpu')))
            
            if return_embedding:
                outcome_sequence_list = np.vstack([outcome_sequence_list, outcome_sequence.cpu().numpy()]) if outcome_sequence_list.size else outcome_sequence.cpu().numpy()
                embedding_outputs_list = np.vstack([embedding_outputs_list, embedding_outputs.cpu().numpy()]) if embedding_outputs_list.size else embedding_outputs.cpu().numpy()
                outcome_label_lists = np.append(outcome_label_lists, outcome_label.cpu().numpy())

    prediction_metrics = get_evaluation_metrics(y_trues, y_preds)

    if verbose: print(f"AUROC:{np.mean(prediction_metrics['AUROC']):.4f}, AUPRC:{np.mean(prediction_metrics['AUPRC']):.4f}")
    
    if return_embedding:
        return prediction_metrics.copy(), outcome_sequence_list, embedding_outputs_list, outcome_label_lists
    else:
        return prediction_metrics.copy()
    

def evaluate_training_temporal(model_scl, model_clf, data_loader, num_classes, device, return_embedding=False, verbose=True, use_nn_clf=False):
    if use_nn_clf:
        return evaluate_training_temporal_nn_clf(model_scl, model_clf, data_loader, num_classes, device, return_embedding, verbose)
    else:
        return evaluate_training_temporal_predictor_clf(model_scl, model_clf, data_loader, num_classes, device, return_embedding, verbose)
    

def evaluate_training_temporal_nn_clf(model_scl, model_clf, data_loader, num_classes, device, return_embedding, verbose):
    with torch.no_grad():                    
        model_scl.eval()
        # model_clf.eval()
        model_scl.to(device)
        
        outcome_sequence_list, embedding_outputs_list, outcome_label_lists, state_label_lists = np.array([]), np.array([]), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        y_preds, y_trues = np.empty((0, num_classes)), np.empty((0))
        train_data_Z, train_data_Y = [], []
        
        for i, (seq, _, _, seq_label, seq_outcome, _) in enumerate(data_loader):
            # Move tensors to the configured device
            outcome_sequence = seq.to(device)
            outcome_label = seq_outcome.to(device)

            # Forward pass
            embedding_outputs = model_scl(outcome_sequence)

            train_data_Z.append(embedding_outputs.cpu().numpy())
            train_data_Y.append(outcome_label.cpu().numpy())
            
            if return_embedding:
                outcome_sequence_list = np.vstack([outcome_sequence_list, outcome_sequence.cpu().numpy()]) if outcome_sequence_list.size else outcome_sequence.cpu().numpy()
                embedding_outputs_list = np.vstack([embedding_outputs_list, embedding_outputs.cpu().numpy()]) if embedding_outputs_list.size else embedding_outputs.cpu().numpy()
                outcome_label_lists = np.append(outcome_label_lists, outcome_label.cpu().numpy())
                state_label_lists = np.append(state_label_lists, seq_label.cpu().numpy())

        training_data_Z = np.concatenate(train_data_Z)
        training_data_Y = np.concatenate(train_data_Y)
        y_preds = model_clf.predict_proba(training_data_Z)
        y_trues = training_data_Y

    prediction_metrics = get_evaluation_metrics(y_trues, y_preds)

    if verbose: print(f"AUROC:{np.mean(prediction_metrics['AUROC']):.4f}, AUPRC:{np.mean(prediction_metrics['AUPRC']):.4f}")
    
    if return_embedding:
        return prediction_metrics.copy(), outcome_sequence_list, embedding_outputs_list, outcome_label_lists, state_label_lists
    else:
        return prediction_metrics.copy()


def evaluate_training_temporal_predictor_clf(model_scl, model_clf, data_loader, num_classes, device, return_embedding=False, verbose=True):
    # Evaluate Model on Test-Set
    with torch.no_grad():                    
        model_scl.eval()
        model_clf.eval()
        model_clf.to(device)
        model_scl.to(device)
        
        outcome_sequence_list, embedding_outputs_list, outcome_label_lists, state_label_lists = np.array([]), np.array([]), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        y_preds, y_trues = torch.empty((0, num_classes)), torch.empty((0))
         
        for i, (seq, _, _, seq_label, seq_outcome, _) in enumerate(data_loader):
            # Move tensors to the configured device
            outcome_sequence = seq.to(device)
            outcome_label = seq_outcome.to(device)

            # Forward pass
            embedding_outputs = model_scl(outcome_sequence)
            outputs = model_clf(embedding_outputs)

            y_pred = outputs.to('cpu')
            sm_layer = nn.Softmax(dim=1)
            y_pred = sm_layer(y_pred)

            y_preds = torch.cat((y_preds, y_pred.clone().detach().to('cpu')))
            y_trues = torch.cat((y_trues, outcome_label.clone().detach().to('cpu')))
            
            if return_embedding:
                outcome_sequence_list = np.vstack([outcome_sequence_list, outcome_sequence.cpu().numpy()]) if outcome_sequence_list.size else outcome_sequence.cpu().numpy()
                embedding_outputs_list = np.vstack([embedding_outputs_list, embedding_outputs.cpu().numpy()]) if embedding_outputs_list.size else embedding_outputs.cpu().numpy()
                outcome_label_lists = np.append(outcome_label_lists, outcome_label.cpu().numpy())
                state_label_lists = np.append(state_label_lists, seq_label.cpu().numpy())

    prediction_metrics = get_evaluation_metrics(y_trues, y_preds)

    if verbose: print(f"AUROC:{np.mean(prediction_metrics['AUROC']):.4f}, AUPRC:{np.mean(prediction_metrics['AUPRC']):.4f}")
    
    if return_embedding:
        return prediction_metrics.copy(), outcome_sequence_list, embedding_outputs_list, outcome_label_lists, state_label_lists
    else:
        return prediction_metrics.copy()
        