from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import datetime

############################################################
################# Dataset Loading Tools ####################
############################################################

def get_data_splits(data_x, data_y, valid_size=0.25, test_size=0.2, seed=1234, is_binary_classification=False, shuffle_data=True, use_external_class=True):
    '''
    Splits data into train, validation and test sets.
    If is_binary_classification is True, then data_y is converted to binary labels. 
        (For cases where data_y is not originally binary, 
         e.g. Have multiple types of mortality being considered the same label).
    If use_external_class is True, then intermediate states are assigned a new class (i.e. external class).
        (otherwise original intermediate labels are used for intermediate states).

    Parameters:
    - data_x: list of numpy arrays, each array is a sequence of covariates for a patient (i.e. feature trajectory shape = (num_timesteps, num_features))
    - data_y: list of numpy arrays, each array is a sequence of labels for a patient     (i.e. label   trajectory shape = (num_timesteps, 1))
    - valid_size: float, size of validation set
    - test_size: float, size of test set
    - seed: int, random seed
    - is_binary_classification: bool, whether to convert labels to binary
    - shuffle_data: bool, whether to shuffle data before splitting
    - use_external_class: bool, whether to use external class for intermediate states

    Returns:
    + tr_data_x, va_data_x, te_data_x: list of numpy arrays for each split (i.e. feature trajectories), each array has shape = (num_timesteps, num_features)
    + tr_data_y, va_data_y, te_data_y: list of numpy arrays for each split (i.e. original label trajectories),   each array has shape = (num_timesteps, 1)
    + tr_labels_outcome, va_labels_outcome, te_labels_outcome: numpy array of the final label of each patient trajectory with shape = (num_patients,)
    + tr_data_y_temporal, va_data_y_temporal, te_data_y_temporal: list of numpy arrays for each split (i.e. label trajectories), 
                                                                  each array has shape = (num_timesteps, 1)
                                                                  If external_class is used, then non-terminal timesteps have external_class
                                                                  otherwise, non-terminal timesteps have original label (same as tr_data_y, va_data_y, te_data_y)
    + list_of_classes, num_classes, external_class

    '''

    tr_data_x,te_data_x, tr_data_y,te_data_y = train_test_split(data_x, data_y, test_size=0.2, random_state=seed, shuffle=shuffle_data)
    tr_data_x,va_data_x, tr_data_y,va_data_y = train_test_split(tr_data_x, tr_data_y, test_size=0.25, random_state=seed, shuffle=shuffle_data)

    tr_labels_outcome = np.array([lb[-1] for lb in tr_data_y]).flatten()
    te_labels_outcome = np.array([lb[-1] for lb in te_data_y]).flatten()
    va_labels_outcome = np.array([lb[-1] for lb in va_data_y]).flatten()

    # Make Labels Binary
    if is_binary_classification:
        tr_data_y = [np.where(d_y>=1, 1, d_y) for d_y in tr_data_y]
        te_data_y = [np.where(d_y>=1, 1, d_y) for d_y in te_data_y]
        va_data_y = [np.where(d_y>=1, 1, d_y) for d_y in va_data_y]
        tr_labels_outcome = np.where(tr_labels_outcome>=1, 1, tr_labels_outcome)
        te_labels_outcome = np.where(te_labels_outcome>=1, 1, te_labels_outcome)
        va_labels_outcome = np.where(va_labels_outcome>=1, 1, va_labels_outcome)
    
    list_of_classes = np.unique(tr_labels_outcome)
    num_classes = len(list_of_classes)

    if use_external_class:
        external_class = max(list_of_classes) + 1  #Class to be assigned to intermediate states
        
        # Create Temporal Labels (i.e. Label for each state in time)
        tr_data_y_temporal = [np.expand_dims(np.append((np.ones(np.shape(d_y))*external_class)[:-1],d_y[-1]), axis=1) for d_y in tr_data_y]
        te_data_y_temporal = [np.expand_dims(np.append((np.ones(np.shape(d_y))*external_class)[:-1],d_y[-1]), axis=1) for d_y in te_data_y]
        va_data_y_temporal = [np.expand_dims(np.append((np.ones(np.shape(d_y))*external_class)[:-1],d_y[-1]), axis=1) for d_y in va_data_y]
    else:
        external_class = None

        # Create Temporal Labels (i.e. Label for each state in time)
        tr_data_y_temporal = [d_y for d_y in tr_data_y]
        te_data_y_temporal = [d_y for d_y in te_data_y]
        va_data_y_temporal = [d_y for d_y in va_data_y]
    
    return tr_data_x, tr_data_y, va_data_x, va_data_y, te_data_x, te_data_y, \
            tr_labels_outcome, va_labels_outcome, te_labels_outcome, \
            tr_data_y_temporal, va_data_y_temporal, te_data_y_temporal, \
            list_of_classes, num_classes, external_class


def get_data_temporal_evaluation(X_temporal, y_temporal, y_outcome):
    X_data_iid  = np.concatenate(X_temporal)
    X_last_T   = np.concatenate([np.tile(x_i[-1], (len(x_i),1)) for x_i in X_temporal])
    state_index = np.concatenate([range(len(x_i)) for x_i in X_temporal])
    state_label = np.concatenate(y_temporal).flatten()
    outcome_lbl = np.concatenate([len(x_i)*[y_outcome[i]] for i, x_i in enumerate(X_temporal)])
    seq_lengths = np.concatenate([len(x_i)*[len(x_i)] for x_i in X_temporal])
    return list(zip(X_data_iid, X_last_T, state_index, state_label, outcome_lbl, seq_lengths))


def get_data_temporal_training(X_temporal, y_temporal, y_outcome, NN_num=1, NN_metric=None, approximate_nn=False, is_ssl=False):
    
    X_data_iid  = np.concatenate(X_temporal)
    X_data_temporal = [x_i[:i+1] for x_i in X_temporal for i in range(len(x_i))]
    X_last_T   = np.concatenate([np.tile(x_i[-1], (len(x_i),1)) for x_i in X_temporal])
    state_index = np.concatenate([range(len(x_i)) for x_i in X_temporal])
    state_label = np.concatenate(y_temporal).flatten()
    outcome_lbl = np.concatenate([len(x_i)*[y_outcome[i]] for i, x_i in enumerate(X_temporal)])
    seq_lengths = np.concatenate([len(x_i)*[len(x_i)] for x_i in X_temporal])

    if NN_metric is None:
        NN_metric = "minkowski"
    
    KNN_num = 2 if NN_num is None else NN_num + 1

    if KNN_num < 2:
        raise ValueError("NN_num must be at least 1.")
    if approximate_nn and KNN_num > 2:
        Warning('WARNING: Will not use fast approximate NN search since KNN_num > 2.')
    
    if approximate_nn:
        return group_batch_based_on_approximate_nn(X_data_iid, X_data_temporal, X_last_T, state_index, state_label, outcome_lbl, seq_lengths, 
                                                   KNN_num, NN_metric, 
                                                   is_ssl)
    else:
        return group_batch_based_on_exact_nn(X_data_iid, X_data_temporal, X_last_T, state_index, state_label, outcome_lbl, seq_lengths, 
                                             KNN_num, NN_metric, 
                                             is_ssl)


def group_batch_based_on_approximate_nn(X_data_iid, X_data_temporal, X_last_T, state_index, state_label, outcome_lbl, seq_lengths, 
                                        KNN_num, NN_metric, 
                                        is_ssl):
    raise NotImplementedError("TODO: Approximate NN search using HNSW not implemented yet.")


def group_batch_based_on_exact_nn(X_data_iid, X_data_temporal, X_last_T, state_index, state_label, outcome_lbl, seq_lengths, 
                                  KNN_num, NN_metric,
                                  is_ssl):
    neigh = NearestNeighbors(n_neighbors=len(X_data_iid), metric=NN_metric)
    neigh.fit(X_data_iid)

    X_data_iid_grp, X_data_temporal_grp, X_last_T_grp, state_index_grp, state_label_grp, outcome_lbl_grp, seq_lengths_grp = [], [], [], [], [], [], []
    list_of_captured_indices = []
    for iid_idx, x_i in enumerate(X_data_iid):
        if iid_idx in list_of_captured_indices: continue
        
        all_neighbors = neigh.kneighbors([x_i], return_distance=False).squeeze()

        # If not SSL, then only consider neighbours with the same outcome label otherwise KNN_num neighbours
        if is_ssl:
            curr_neighbours = list(np.arange(min(KNN_num, len(all_neighbors))).astype(int))
        else:
            curr_neighbours = []
            for nei_tmp in all_neighbors:
                if len(curr_neighbours) == KNN_num: 
                    break
                # elif outcome_lbl[nei_tmp] == outcome_lbl[iid_idx]:
                elif state_label[nei_tmp] == state_label[iid_idx]:
                    curr_neighbours.append(int(nei_tmp))

        X_data_iid_grp.append(X_data_iid[curr_neighbours])
        X_data_temporal_grp.append([X_data_temporal[nei_idx] for nei_idx in curr_neighbours])
        X_last_T_grp.append(X_last_T[curr_neighbours])
        state_index_grp.append(state_index[curr_neighbours])
        state_label_grp.append(state_label[curr_neighbours])
        outcome_lbl_grp.append(outcome_lbl[curr_neighbours])
        seq_lengths_grp.append(seq_lengths[curr_neighbours])

        # if not np.all(outcome_lbl[curr_neighbours] == outcome_lbl[curr_neighbours][0]):
        if not np.all(state_label[curr_neighbours] == state_label[curr_neighbours][0]):
            print(np.all(state_label[curr_neighbours]))

        list_of_captured_indices.extend(curr_neighbours)
    
    return list(zip(X_data_iid_grp, X_last_T_grp, state_index_grp, state_label_grp, outcome_lbl_grp, seq_lengths_grp, X_data_temporal_grp)), outcome_lbl_grp


def get_data_temporal_training_no_nn(X_temporal, y_temporal, y_outcome):
    X_data_iid  = np.concatenate(X_temporal)
    X_data_temporal = [np.array(x_i[:i+1]) for x_i in X_temporal for i in range(len(x_i))]
    X_last_T   = np.concatenate([np.tile(x_i[-1], (len(x_i),1)) for x_i in X_temporal])
    state_index = np.concatenate([range(len(x_i)) for x_i in X_temporal])
    state_label = np.concatenate(y_temporal).flatten()
    outcome_lbl = np.concatenate([len(x_i)*[y_outcome[i]] for i, x_i in enumerate(X_temporal)])
    seq_lengths = np.concatenate([len(x_i)*[len(x_i)] for x_i in X_temporal])

    return list(zip(X_data_iid, X_last_T, state_index, state_label, outcome_lbl, seq_lengths, X_data_temporal)), outcome_lbl


def get_sample_weights(labels_outcome):
    '''
    Generates a sampler with weights for each sample in a dataset based on the outcome labels (addressing class imbalance).

    Parameters:
    - labels_outcome (array-like): A list or numpy array of outcome labels for each sample in the dataset.

    Returns:
    + sampler (WeightedRandomSampler): A sampler that can be used with a DataLoader to draw samples with the specified probabilities.
    '''
    _, count_per_class = np.unique(labels_outcome, return_counts=True)
    count_per_class = [len(labels_outcome)/count_per_class[i] for i in range(len(count_per_class))]
    sample_weights = np.array([count_per_class[int(lbl)] for lbl in labels_outcome])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        self.batch_size = batch_size
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(np.ceil(len(y)/self.batch_size))
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return int(np.ceil(len(self.y)/self.batch_size)) 


def collate_fn_pretrain_timestep(batch):
    '''
    Custom collate function for preprocessing a batch of outcome sequences for pretraining.
    It extracts the last timestep from each sequence in the batch and compiles them into a single tensor 
    along with its corresponding ourcome label. 

    Parameters:
    - batch (list of tuples): A list where each tuple contains a sequence tensor and a label. 
      The sequence tensor is expected to be of shape (seq_length, feature_size), where seq_length 
      is the length of the sequence, and feature_size is the size of each feature vector.
    
    Returns:
    + tuple: A tuple containing two elements:
        + seq_lasts_list (torch.Tensor): A tensor of shape (batch_size, feature_size) containing 
          the last timestep features from each sequence in the batch.
        + label_tensor (torch.Tensor): A tensor of shape (batch_size,) containing the labels (outcome) of the sequences.
    '''
    seq_lasts_list = torch.empty(size=(0,batch[0][0].shape[-1]))
    label_list = []
    
    for (_seq,_label) in batch:
        seq_lasts_list = torch.cat((seq_lasts_list, torch.as_tensor([_seq[-1]]).float()), dim=0)
        label_list.append(int(_label))
    
    return seq_lasts_list, torch.tensor(label_list, dtype=torch.long)


def collate_fn_all_timesteps_evaluation(batch):    
    seq_batch_list, seq_last_timestep_batch_list = torch.empty(size=(0,batch[0][0].shape[0])), torch.empty(size=(0,batch[0][0].shape[0]))
    state_idx_list, state_label_list, outcome_label_list, seq_lens_list = [], [], [], []
    
    for (_seq, _seq_last_timestep, _state_idx, _state_label, _outcome_label, _seq_len) in batch:
        seq_batch_list = torch.cat((seq_batch_list, torch.as_tensor(_seq).float().unsqueeze(dim=0)), dim=0)
        seq_last_timestep_batch_list = torch.cat((seq_last_timestep_batch_list, torch.as_tensor(_seq_last_timestep).float().unsqueeze(dim=0)), dim=0)
        state_idx_list.append(int(_state_idx))
        state_label_list.append(int(_state_label))
        outcome_label_list.append(int(_outcome_label))
        seq_lens_list.append(int(_seq_len))
    
    return seq_batch_list, seq_last_timestep_batch_list, torch.tensor(state_idx_list, dtype=torch.long), torch.tensor(state_label_list, dtype=torch.long), \
            torch.tensor(outcome_label_list, dtype=torch.long), torch.tensor(seq_lens_list, dtype=torch.long)


def collate_fn_all_timesteps_training(batch):    
    seq_batch_list, seq_last_timestep_batch_list = torch.empty(size=(0,batch[0][0].shape[1])), torch.empty(size=(0,batch[0][0].shape[1]))
    state_idx_list, state_label_list, outcome_label_list, seq_lens_list = [], [], [], []
    seq_history_batch_list_x, seq_history_batch_list_y = [], torch.empty(size=(0,batch[0][0].shape[1]))
    seq_pair_assignment = []
    
    for batch_idx, (_seq, _seq_last_timestep, _state_idx, _state_label, _outcome_label, _seq_len, _seq_with_history) in enumerate(batch):
        # pdb.set_trace()
        seq_batch_list = torch.cat((seq_batch_list, torch.as_tensor(_seq).float()), dim=0)
        seq_last_timestep_batch_list = torch.cat((seq_last_timestep_batch_list, torch.as_tensor(_seq_last_timestep).float()), dim=0)
        state_idx_list.extend(_state_idx)
        state_label_list.extend([int(st_lbl) for st_lbl in _state_label])
        outcome_label_list.extend([int(ot_lbl) for ot_lbl in _outcome_label])
        seq_lens_list.extend(_seq_len)
        seq_history_batch_list_x.extend([torch.as_tensor(seq_hist_curr[:-1]).float() for seq_hist_curr in _seq_with_history])
        seq_history_batch_list_y = torch.cat((seq_history_batch_list_y, torch.stack([torch.as_tensor(seq_hist_curr[-1]).float()  for seq_hist_curr in _seq_with_history])), dim=0)
        seq_pair_assignment.extend([batch_idx]*_seq.shape[0])
    
    batch_seq_lengths = torch.tensor([s.shape[0] for s in seq_history_batch_list_x])
    seq_history_batch_list_x = torch.nn.utils.rnn.pad_sequence(seq_history_batch_list_x, batch_first=True, padding_value=0.0)
    batch_seq_lengths_sorted, batch_seq_lengths_sorted_idx = np.sort(batch_seq_lengths)[::-1], np.argsort(-batch_seq_lengths)
    batch_seq_hist_x = seq_history_batch_list_x[batch_seq_lengths_sorted_idx]
    batch_seq_hist_y = seq_history_batch_list_y[batch_seq_lengths_sorted_idx]

    seq_batch_list = seq_batch_list[batch_seq_lengths_sorted_idx]
    seq_last_timestep_batch_list = seq_last_timestep_batch_list[batch_seq_lengths_sorted_idx]
    state_idx_list = torch.tensor(state_idx_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    state_label_list = torch.tensor(state_label_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    outcome_label_list = torch.tensor(outcome_label_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    seq_lens_list = torch.tensor(seq_lens_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    seq_pair_assignment = torch.tensor(seq_pair_assignment, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    
    return seq_batch_list, seq_last_timestep_batch_list, state_idx_list, state_label_list, \
            outcome_label_list, seq_lens_list, \
            ((batch_seq_hist_x, batch_seq_hist_y), torch.tensor(batch_seq_lengths_sorted.copy())) , seq_pair_assignment


def collate_fn_all_timesteps_training_no_nn(batch):    
    seq_batch_list, seq_last_timestep_batch_list = torch.empty(size=(0,batch[0][0].shape[0])), torch.empty(size=(0,batch[0][0].shape[0]))
    state_idx_list, state_label_list, outcome_label_list, seq_lens_list = [], [], [], []
    seq_history_batch_list_x, seq_history_batch_list_y = [], torch.empty(size=(0,batch[0][0].shape[0]))
    seq_pair_assignment = []
    
    for batch_idx, (_seq, _seq_last_timestep, _state_idx, _state_label, _outcome_label, _seq_len, _seq_with_history) in enumerate(batch):
        # pdb.set_trace()
        seq_batch_list = torch.cat((seq_batch_list, torch.as_tensor(_seq).float().unsqueeze(dim=0)), dim=0)
        seq_last_timestep_batch_list = torch.cat((seq_last_timestep_batch_list, torch.as_tensor(_seq_last_timestep).float().unsqueeze(dim=0)), dim=0)
        state_idx_list.append(int(_state_idx))
        state_label_list.append(int(_state_label))
        outcome_label_list.append(int(_outcome_label))
        seq_lens_list.append(int(_seq_len))
        
        # seq_history_batch_list_x.extend([torch.as_tensor(seq_hist_curr[:-1]).float() for seq_hist_curr in _seq_with_history])
        seq_history_batch_list_x.append(torch.as_tensor(_seq_with_history[:-1]).float())
        seq_history_batch_list_y = torch.cat((seq_history_batch_list_y, torch.as_tensor(_seq_with_history[-1]).float().unsqueeze(dim=0)), dim=0)
        seq_pair_assignment.extend([batch_idx]*_seq.shape[0])

    batch_seq_lengths = torch.tensor([s.shape[0] for s in seq_history_batch_list_x])
    seq_history_batch_list_x = torch.nn.utils.rnn.pad_sequence(seq_history_batch_list_x, batch_first=True, padding_value=0.0)
    batch_seq_lengths_sorted, batch_seq_lengths_sorted_idx = np.sort(batch_seq_lengths)[::-1], np.argsort(-batch_seq_lengths)
    batch_seq_hist_x = seq_history_batch_list_x[batch_seq_lengths_sorted_idx]
    batch_seq_hist_y = seq_history_batch_list_y[batch_seq_lengths_sorted_idx]

    seq_batch_list = seq_batch_list[batch_seq_lengths_sorted_idx]
    seq_last_timestep_batch_list = seq_last_timestep_batch_list[batch_seq_lengths_sorted_idx]
    state_idx_list = torch.tensor(state_idx_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    state_label_list = torch.tensor(state_label_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    outcome_label_list = torch.tensor(outcome_label_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    seq_lens_list = torch.tensor(seq_lens_list, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    seq_pair_assignment = torch.tensor(seq_pair_assignment, dtype=torch.long)[batch_seq_lengths_sorted_idx]
    
    return seq_batch_list, seq_last_timestep_batch_list, state_idx_list, state_label_list, \
           outcome_label_list, seq_lens_list, \
            ((batch_seq_hist_x, batch_seq_hist_y), torch.tensor(batch_seq_lengths_sorted.copy())) , seq_pair_assignment


############################################################
#################### Saving Models #########################
############################################################


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        print("Model not saved. No path provided.")
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)


def load_checkpoint(load_path, model, optimizer, device='cpu'):
    if load_path==None:
        print("Model not Loaded. No path provided.")
        return None
    
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def get_tensorboard_namer(experiment_name, is_binary_classification, is_hyper_sphere, hidden_sizes_scl, embedding_size,
                          activation_scl, activation_clf,
                          train_batch_size, lr_scl, lr_clf, lr_temporal,
                          temperature_scl, alpha_temporal, temporal_l2_reg_coeff,
                          do_nn_pairing, nn_pairing_k,
                          use_nn_clf, model_clf_str, knn_clf, hidden_sizes_clf, 
                          temporal_network, temporal_network_embedding_size, temporal_network_num_layers,
                          tensorboard_suffix=None, include_datetime=True):
    
    experiment_runtime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    tensorboard_string_extension = "TSCL_" + experiment_name
    tensorboard_string_extension += "__binary__" if is_binary_classification else "__"
    tensorboard_string_extension += "HyperSphere__" if is_hyper_sphere else "Euclidean__"
    tensorboard_string_extension += "__".join([f"hsz{hsz_i_}_{hsz_}" for hsz_i_, hsz_ in enumerate(hidden_sizes_scl)]) + "__" + f"esz_{embedding_size}__"
    tensorboard_string_extension += f"actSCL_{activation_scl}_actCLF_{activation_clf}__"
    tensorboard_string_extension += f"bsz_{train_batch_size}__lrSCL_{lr_scl}__lrCLF_{lr_clf}__lrTEMP_{lr_temporal}__"
    tensorboard_string_extension += f"tmpSCL_{temperature_scl}__alphaTEMP_{alpha_temporal}__l2regTEMP_{temporal_l2_reg_coeff}__"
    tensorboard_string_extension += f"doNNP_{do_nn_pairing}__kNNP_{nn_pairing_k}__"
    tensorboard_string_extension += f"_nnCLF_{model_clf_str}__" if use_nn_clf else f"predCLF_{'_'.join([f'hszClf{hsz_i_}_{hsz_}' for hsz_i_, hsz_ in enumerate(hidden_sizes_clf)])}__"
    tensorboard_string_extension += f"temporalNetwork_{temporal_network}__" if temporal_network is not None else ""
    tensorboard_string_extension += f"temporalHSZ_{temporal_network_embedding_size}__temporalLayers_{temporal_network_num_layers}" if temporal_network is not None else ""
    tensorboard_string_extension = (tensorboard_string_extension + "__" + tensorboard_suffix) if  (tensorboard_suffix is not None) else tensorboard_string_extension
    tensorboard_string_extension += f"_{experiment_runtime}" if include_datetime else ""

    return tensorboard_string_extension