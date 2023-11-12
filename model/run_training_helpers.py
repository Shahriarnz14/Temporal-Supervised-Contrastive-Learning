from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from utils.utils import *
from temporal_scl_model import TSCL_Encoder_Network, TSL_Predictor_Network, TSCL_Temporal_Network
from evaluate import evaluate_training, evaluate_training_temporal
from pretrain import train_temporal_scl_pretrain_phase
from train import train_temporal_SCL_main_phase
import pickle, os


def run_temporal_scl(embedding_size, hidden_sizes_scl, hidden_sizes_clf, hidden_size_temporal, num_temporal_layers, activation_scl, activation_clf, 
                    is_hyper_sphere, is_binary_classification,
                    train_batch_size, valid_batch_size, test_batch_size,
                    num_epochs_scl, num_epochs_clf, num_epochs_temporal,
                    num_epochs_scl_pretrain, num_epochs_clf_pretrain,
                    lr_scl, lr_clf, lr_temporal,
                    temperature_scl,
                    model_clf_str, model_temporal_str, criterion_temporal,
                    do_nn_pairing, nn_pairing_k,
                    alpha_temporal, temporal_l2_reg_coeff,
                    data_x, data_y, use_external_class,
                    predict_raw_temporal,
                    valid_size, test_size, seed, shuffle_data,
                    return_embedding=False, use_gpu=True,
                    use_nn_clf=False, knn_clf=2, nn_metric=None, approximate_nn=False,
                    saved_train_loader=False, do_save_train_loader=False, nn_train_loader_path=None,
                    tensorboar_string="TSCL", experiment_name='synthetic', verbose=True):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if verbose: print("Pretraining Temporal-SCL:")    
    pretraining_results_dict = run_temporal_scl_pretraining(embedding_size=embedding_size, hidden_sizes_scl=hidden_sizes_scl, hidden_sizes_clf=hidden_sizes_clf,
                                                            activation_scl=activation_scl, activation_clf=activation_clf,
                                                            is_hyper_sphere=is_hyper_sphere, is_binary_classification=is_binary_classification,
                                                            train_batch_size=train_batch_size, valid_batch_size=valid_batch_size, test_batch_size=test_batch_size,
                                                            num_epochs_scl=num_epochs_scl_pretrain, num_epochs_clf=num_epochs_clf_pretrain,
                                                            lr_scl=lr_scl, lr_clf=lr_clf,
                                                            temperature_scl=temperature_scl,
                                                            data_x=data_x, data_y=data_y, use_external_class=use_external_class,
                                                            valid_size=valid_size, test_size=test_size, seed=seed, shuffle_data=shuffle_data,
                                                            return_embedding=return_embedding, use_gpu=use_gpu,
                                                            tensorboar_string=tensorboar_string, experiment_name=experiment_name, verbose=verbose)
    
    model_scl = pretraining_results_dict["models"]["scl"]
    if verbose: print("Pretrained Temporal-SCL.")

    if saved_train_loader and nn_train_loader_path is not None:
        try:
            print("Trying to load saved train loader from: " + nn_train_loader_path)
            with open(nn_train_loader_path, "rb") as handle:
                saved_train_loader = pickle.load(handle)
            print("Successfully loaded saved train loader.")
        except OSError:
            Warning(f"Could not load saved train loader from {nn_train_loader_path}. Will recompute it.")
            Warning("This will take some time. It will be saved in: " + f"../results/{experiment_name}/{experiment_name}_train_{nn_pairing_k}NN_loader.pkl")
            saved_train_loader = False
    else:
        print("No saved train loader was provided. Will recompute it.")
    
    if verbose: print("Training Temporal-SCL:")
    training_results_dict = run_temporal_scl_training(embedding_size=embedding_size, hidden_sizes_clf=hidden_sizes_clf,
                                                      hidden_size_temporal=hidden_size_temporal, num_temporal_layers=num_temporal_layers,
                                                      activation_clf=activation_clf,
                                                      model_scl=model_scl, model_clf_str=model_clf_str, model_temporal_str=model_temporal_str,
                                                      is_binary_classification=is_binary_classification,
                                                      train_batch_size=train_batch_size, valid_batch_size=valid_batch_size, test_batch_size=test_batch_size,
                                                      num_epochs_scl=num_epochs_scl, num_epochs_clf=num_epochs_clf, num_epochs_temporal=num_epochs_temporal,
                                                      lr_scl=lr_scl, lr_clf=lr_clf, lr_temporal=lr_temporal,
                                                      temperature_scl=temperature_scl,
                                                      do_nn_pairing=do_nn_pairing, nn_pairing_k=nn_pairing_k,
                                                      alpha_temporal=alpha_temporal, temporal_l2_reg_coeff=temporal_l2_reg_coeff,
                                                      data_x=data_x, data_y=data_y, use_external_class=use_external_class, predict_raw_temporal=predict_raw_temporal,
                                                      criterion_temporal=criterion_temporal,
                                                      valid_size=valid_size, test_size=test_size, seed=seed, shuffle_data=shuffle_data,
                                                      use_nn_clf=use_nn_clf, knn_clf=knn_clf, nn_metric=nn_metric, approximate_nn=approximate_nn,
                                                      return_embedding=return_embedding, saved_train_loader=saved_train_loader, 
                                                      do_save_train_loader=do_save_train_loader, use_gpu=use_gpu,
                                                      tensorboar_string=tensorboar_string, experiment_name=experiment_name, verbose=verbose)
    if verbose: print("Trained Temporal-SCL.")

    return pretraining_results_dict, training_results_dict
    


def run_temporal_scl_pretraining(embedding_size, hidden_sizes_scl, hidden_sizes_clf, activation_scl, activation_clf, 
                                 is_hyper_sphere, is_binary_classification,
                                 train_batch_size, valid_batch_size, test_batch_size,
                                 num_epochs_scl, num_epochs_clf,
                                 lr_scl, lr_clf,
                                 temperature_scl,
                                 data_x, data_y, use_external_class,
                                 valid_size, test_size, seed, shuffle_data,
                                 return_embedding=False, use_gpu=True,
                                 tensorboar_string="TSCL", experiment_name='synthetic', verbose=True):

    (tr_data_x, _, va_data_x, _, te_data_x, _, 
     tr_labels_outcome, va_labels_outcome, te_labels_outcome, 
     _, _, _, 
     _, num_classes, _) \
        = get_data_splits(data_x=data_x, data_y=data_y, 
                          is_binary_classification=is_binary_classification,
                          valid_size=valid_size, test_size=test_size,  
                          seed=seed, shuffle_data=shuffle_data, use_external_class=use_external_class)
    
    dl_train = DataLoader(list(zip(tr_data_x, tr_labels_outcome)), batch_size=train_batch_size, 
                          collate_fn=collate_fn_pretrain_timestep, sampler=get_sample_weights(tr_labels_outcome))
    dl_valid = DataLoader(list(zip(va_data_x, va_labels_outcome)), batch_size=valid_batch_size, 
                          collate_fn=collate_fn_pretrain_timestep, sampler=get_sample_weights(va_labels_outcome))
    dl_test  = DataLoader(list(zip(te_data_x, te_labels_outcome)), shuffle=True, batch_size=test_batch_size,  
                          collate_fn=collate_fn_pretrain_timestep)
    

    # Check Device configuration
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if verbose: print(f"Device: {device}")

    # Define model
    input_size = tr_data_x[0].shape[-1] #55
    model_scl = TSCL_Encoder_Network(input_size, hidden_sizes_scl, embedding_size, activation_scl, is_hyper_sphere).to(device)
    model_clf = TSL_Predictor_Network(embedding_size, num_classes, hidden_sizes_clf, activation_clf)
    train_temporal_scl_pretrain_phase(model_scl=model_scl, model_clf=model_clf, 
                                      train_loader=dl_train, valid_loader=dl_valid,
                                      num_epochs_scl=num_epochs_scl, num_epochs_clf=num_epochs_clf,
                                      temperature_scl=temperature_scl,
                                      lr_scl=lr_scl, lr_clf=lr_clf, device=device,
                                      tensorboard_filename=tensorboar_string, experiment_name=experiment_name, verbose=verbose)
    
    
    if verbose: print("Pretrained Temporal-SCL - Training:", end=" ") 
    pred_result_train = evaluate_training(model_scl, model_clf, dl_train, num_classes, device, verbose=verbose)
    if verbose: print("Pretrained Temporal-SCL - Validation:", end=" ") 
    pred_result_valid = evaluate_training(model_scl, model_clf, dl_valid, num_classes, device, verbose=verbose)
    if verbose: print("Pretrained Temporal-SCL - Test:", end=" ") 
    pred_result_test  = evaluate_training(model_scl, model_clf, dl_test, num_classes, device, verbose=verbose)
    
    return_dict = {"models":{"scl":model_scl, "clf": model_clf}, 
                   "prediction_results":{"Train": pred_result_train, "Valid": pred_result_valid, "Test": pred_result_test}, 
                   "tensorboard_str":tensorboar_string}
    if return_embedding:
        dl_train_full  = DataLoader(list(zip(tr_data_x, tr_labels_outcome)), shuffle=True, batch_size=len(tr_data_x),  collate_fn=collate_fn_pretrain_timestep)
        dl_valid_full  = DataLoader(list(zip(va_data_x, va_labels_outcome)), shuffle=True, batch_size=len(va_data_x),  collate_fn=collate_fn_pretrain_timestep)
        dl_test_full   = DataLoader(list(zip(te_data_x, te_labels_outcome)), shuffle=True, batch_size=len(te_data_x),  collate_fn=collate_fn_pretrain_timestep)
        dataset_str = ["temporal_scl_pretrained_train_embedding", "temporal_scl_pretrained_valid_embedding", "temporal_scl_pretrained_test_embedding"]
        for i_dataset, dataset_current in enumerate([dl_train_full, dl_valid_full, dl_test_full]):
            _, outcome_sequence_, embedding_outputs_, outcome_label_ = evaluate_training(model_scl, model_clf, dataset_current, num_classes, device, 
                                                                                         return_embedding=True, verbose=False)
            return_dict[dataset_str[i_dataset]] =  {"input_seq": outcome_sequence_, "embedding_seq": embedding_outputs_, "labels_outcome": outcome_label_}
    return return_dict


def run_temporal_scl_training(embedding_size, hidden_sizes_clf, hidden_size_temporal, num_temporal_layers,
                              activation_clf, 
                              model_scl, model_clf_str, model_temporal_str,
                              is_binary_classification,
                              train_batch_size, valid_batch_size, test_batch_size,
                              num_epochs_scl, num_epochs_clf, num_epochs_temporal,
                              lr_scl, lr_clf, lr_temporal,
                              temperature_scl,
                              do_nn_pairing, nn_pairing_k,
                              alpha_temporal, temporal_l2_reg_coeff,
                              data_x, data_y, use_external_class, predict_raw_temporal,
                              criterion_temporal,
                              valid_size, test_size, seed, shuffle_data=True,
                              use_nn_clf=False, knn_clf=2, nn_metric=None, approximate_nn=False,
                              return_embedding=False, saved_train_loader=False, do_save_train_loader=False, use_gpu=True,
                              tensorboar_string="TSCL", experiment_name='synthetic', verbose=True):
    
    (tr_data_x, _, va_data_x, _, te_data_x, _, 
     tr_labels_outcome, va_labels_outcome, te_labels_outcome, 
     tr_data_y_temporal, va_data_y_temporal, te_data_y_temporal, 
     _, num_classes, _) \
        = get_data_splits(data_x, data_y, 
                          is_binary_classification=is_binary_classification,
                          valid_size=valid_size, test_size=test_size, 
                          seed=seed, shuffle_data=shuffle_data, use_external_class=use_external_class)
    
    
    
    # Check Device configuration
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    if verbose: print(f"Device: {device}")

    ## Temporal Data Loaders   
    if do_nn_pairing is False:
        tr_data_temporal, tr_outcome_temporal = get_data_temporal_training_no_nn(tr_data_x, tr_data_y_temporal, tr_labels_outcome)

        dl_train_temporal = DataLoader(dataset=tr_data_temporal, 
                                       batch_sampler=StratifiedBatchSampler(np.array(tr_outcome_temporal), batch_size=train_batch_size, shuffle=shuffle_data), 
                                       collate_fn=collate_fn_all_timesteps_training_no_nn)
        
    else:
        KNN_num = nn_pairing_k + 1 if do_nn_pairing is not None else 2

        # If already saved from before in saved_train_loader dictionary load it
        if saved_train_loader is not False:
            tr_data_temporal, tr_outcome_temporal = saved_train_loader["dataset"], saved_train_loader["outcome_temporal"]
        else:
            tr_data_temporal, tr_outcome_temporal = get_data_temporal_training(X_temporal=tr_data_x, y_temporal=tr_data_y_temporal, y_outcome=tr_labels_outcome, 
                                                                               NN_num=nn_pairing_k, NN_metric=nn_metric, approximate_nn=approximate_nn, is_ssl=False)
            saved_train_loader = {"dataset": tr_data_temporal, "outcome_temporal": tr_outcome_temporal}
            if do_save_train_loader:
                # check if results folders and the corresponding paths exist
                results_path = f"../results/{experiment_name}/"
                if not os.path.exists(results_path):
                    os.makedirs(results_path)
                with open(f"{results_path}{experiment_name}_train_{KNN_num-1}NN_loader.pkl", "wb") as handle:
                    pickle.dump(saved_train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        dl_train_temporal = DataLoader(dataset=tr_data_temporal, 
                                       batch_sampler=StratifiedBatchSampler(np.array(tr_outcome_temporal)[:,0], batch_size=int(train_batch_size/KNN_num), shuffle=shuffle_data), 
                                       collate_fn=collate_fn_all_timesteps_training)
    dl_valid_temporal = DataLoader(dataset=get_data_temporal_evaluation(va_data_x, va_data_y_temporal, va_labels_outcome), 
                                   batch_sampler=StratifiedBatchSampler(np.concatenate(va_data_y_temporal).flatten(), batch_size=valid_batch_size, shuffle=shuffle_data), 
                                   collate_fn=collate_fn_all_timesteps_evaluation)
    dl_test_temporal  = DataLoader(dataset=get_data_temporal_evaluation(te_data_x, te_data_y_temporal, te_labels_outcome), 
                                   batch_sampler=StratifiedBatchSampler(np.concatenate(te_data_y_temporal).flatten(), batch_size=test_batch_size , shuffle=shuffle_data), 
                                   collate_fn=collate_fn_all_timesteps_evaluation)
    

    if use_nn_clf: 
        if model_clf_str == "KNN":
            model_clf = KNeighborsClassifier(n_neighbors=knn_clf, n_jobs=-1)
        elif model_clf_str == "SVM":
            model_clf = SVC(probability=True)
        elif model_clf_str == "RF":
            model_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif model_clf_str == "LR":
            model_clf = LogisticRegression(n_jobs=-1)
        elif model_clf_str == "MLP":
            Warning("Classic Predictor was set to 'True' for the Predictor Head of Temporal-SCL, but MLP was selected as the model. MLP is not a classic predictor.")
            model_clf = TSL_Predictor_Network(embedding_size, num_classes, hidden_sizes_clf, activation_clf)
        else:
            raise ValueError(f"{model_clf_str} for Predictor Head is not recognized")
    else:
        model_clf = TSL_Predictor_Network(embedding_size, num_classes, hidden_sizes_clf, activation_clf)

    output_temporal_size = tr_data_x[0].shape[-1] if predict_raw_temporal else embedding_size
    model_temporal = TSCL_Temporal_Network(n_features=embedding_size, dimension=hidden_size_temporal, 
                                           num_rnn_layers=num_temporal_layers, rnn_type=model_temporal_str, 
                                           output_dim=output_temporal_size)
    
    train_temporal_SCL_main_phase(model_scl=model_scl, model_clf=model_clf, model_temporal=model_temporal,
                                  train_loader=dl_train_temporal, valid_loader=dl_valid_temporal, num_classes=num_classes, device=device,
                                  num_epochs_scl=num_epochs_scl, num_epochs_clf=num_epochs_clf, num_epochs_temporal=num_epochs_temporal,
                                  temperature_scl=temperature_scl,
                                  lr_scl=lr_scl, lr_clf=lr_clf, lr_temporal=lr_temporal,
                                  do_nn_pairing=do_nn_pairing, nn_pairing_k=nn_pairing_k,
                                  criterion_temporal=criterion_temporal, alpha_temporal=alpha_temporal, 
                                  temporal_l2_reg_coeff=temporal_l2_reg_coeff, predict_raw_temporal=predict_raw_temporal,
                                  use_nn_clf=use_nn_clf, knn_clf=knn_clf,
                                  tensorboard_filename=tensorboar_string, experiment_name=experiment_name, verbose=verbose)    
    
    dl_train_temporal = DataLoader(dataset=get_data_temporal_evaluation(tr_data_x, tr_data_y_temporal, tr_labels_outcome), 
                                   batch_sampler=StratifiedBatchSampler(np.concatenate(tr_data_y_temporal).flatten(), batch_size=train_batch_size, shuffle=shuffle_data), 
                                   collate_fn=collate_fn_all_timesteps_evaluation)
    
    print("Temporal-SCL - Training:", end=" ") 
    pred_result_train = evaluate_training_temporal(model_scl, model_clf, dl_train_temporal, num_classes, device, use_nn_clf=use_nn_clf)
    print("Temporal-SCL - Validation:", end=" ") 
    pred_result_valid = evaluate_training_temporal(model_scl, model_clf, dl_valid_temporal, num_classes, device, use_nn_clf=use_nn_clf)
    print("Temporal-SCL - Test:", end=" ") 
    pred_result_test = evaluate_training_temporal(model_scl, model_clf, dl_test_temporal, num_classes, device, use_nn_clf=use_nn_clf)
    
    return_dict = {"models":{"scl":model_scl, "clf": model_clf, "temporal": model_temporal}, 
                   "prediction_results":{"Train": pred_result_train, "Valid": pred_result_valid, "Test": pred_result_test}, 
                   "tensorboard_str":tensorboar_string}
    
    if return_embedding:
        dataset_str = ["temporal_scl_trained_train_embedding", "temporal_scl_trained_valid_embedding", "temporal_scl_trained_test_embedding"]
        for i_dataset, dataset_current in enumerate([dl_train_temporal, dl_valid_temporal, dl_test_temporal]):
            _, outcome_sequence_, embedding_outputs_, outcome_label_, state_label_ = evaluate_training_temporal(model_scl, model_clf, dataset_current, num_classes, device, 
                                                                                                                return_embedding=True, verbose=False, use_nn_clf=use_nn_clf)
            return_dict[dataset_str[i_dataset]] = {"input_seq": outcome_sequence_, 
                                                   "embedding_seq": embedding_outputs_, 
                                                   "labels_outcome": outcome_label_, 
                                                   "labels_state": state_label_}
    return return_dict