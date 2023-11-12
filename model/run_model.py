import os, pickle
from run_training_helpers import run_temporal_scl
from utils.utils import get_tensorboard_namer


import argparse
import json

def read_args_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # # Implement conditional logic here
    # if not data.get('is_data_splitted', False):
    #     data['is_data_splitted'] = 'False value alternative string'

    # Rest of the function remains the same
    parser = argparse.ArgumentParser(description="Arguments from JSON file")
    for key, value in data.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))

    args = parser.parse_args([])
    return args

def main():
    parser = argparse.ArgumentParser(description="Training and evaluation script for Temporal-SCL")

    # Option to input JSON file path
    parser.add_argument('--config', type=str, help='Path to JSON config file')

    # Direct arguments from args.json
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--embedding_size', type=int, help='Size of the embedding', default=32)
    parser.add_argument('--hidden_sizes_scl', nargs='+', type=int, help='Hidden sizes for scl', default=[32])
    parser.add_argument('--activation_scl', type=str, help='Activation function for scl', default='ReLU')
    parser.add_argument('--is_hyper_sphere', type=bool, help='Is hyper sphere', default=True)
    parser.add_argument('--num_epochs_scl', type=int, help='Number of epochs for scl', default=100)
    parser.add_argument('--lr_scl', type=float, help='Learning rate for scl', default=1e-4)
    parser.add_argument('--model_clf_str', type=str, help='Model clf string', default='MLP')
    parser.add_argument('--hidden_sizes_clf', nargs='+', type=int, help='Hidden sizes for clf', default=[64])
    parser.add_argument('--activation_clf', type=str, help='Activation function for clf', default='ReLU')
    parser.add_argument('--num_epochs_clf', type=int, help='Number of epochs for clf', default=100)
    parser.add_argument('--lr_clf', type=float, help='Learning rate for clf', default=1e-3)
    parser.add_argument('--use_nn_clf', type=bool, help='Use neural network classifier', default=False)
    parser.add_argument('--knn_clf', type=int, help='Knn for classifier', default=2)
    parser.add_argument('--model_temporal_str', type=str, help='Temporal model string', default='LSTM')
    parser.add_argument('--hidden_size_temporal', nargs='+', type=int, help='Hidden size for temporal model', default=[128])
    parser.add_argument('--num_temporal_layers', type=int, help='Number of temporal layers', default=1)
    parser.add_argument('--predict_raw_temporal', type=bool, help='Predict raw temporal', default=False)
    parser.add_argument('--num_epochs_temporal', type=int, help='Number of epochs for temporal', default=10)
    parser.add_argument('--lr_temporal', type=float, help='Learning rate for temporal', default=1e-3)
    parser.add_argument('--criterion_temporal', type=str, help='Criterion for temporal', default='MSE')
    parser.add_argument('--do_nn_pairing', type=bool, help='Do nearest neighbor pairing', default=True)
    parser.add_argument('--nn_pairing_k', type=int, help='Nearest neighbor pairing k', default=5)
    parser.add_argument('--approximate_nn', type=bool, help='Use approximate nearest neighbor', default=False)
    parser.add_argument('--nn_metric', type=str, help='Metric for nearest neighbor', default='minkowski')
    parser.add_argument('--num_epochs_scl_pretrain', type=int, help='Number of epochs for scl pretrain', default=20)
    parser.add_argument('--num_epochs_clf_pretrain', type=int, help='Number of epochs for clf pretrain', default=20)
    parser.add_argument('--temperature_scl', type=float, help='Temperature for scl', default=0.1)
    parser.add_argument('--alpha_temporal', type=int, help='Alpha for temporal', default=50)
    parser.add_argument('--temporal_l2_reg_coeff', type=float, help='Temporal L2 regularization coefficient', default=None)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--shuffle_data', type=bool, help='Shuffle data', default=True)
    parser.add_argument('--train_batch_size', type=int, help='Train batch size', default=256)
    parser.add_argument('--valid_batch_size', type=int, help='Validation batch size', default=256)
    parser.add_argument('--valid_size', type=float, help='Validation size', default=0.25)
    parser.add_argument('--test_batch_size', type=int, help='Test batch size', default=256)
    parser.add_argument('--test_size', type=float, help='Test size', default=0.2)
    parser.add_argument('--is_binary_classification', type=bool, help='Is binary classification', default=False)
    parser.add_argument('--use_external_class', type=bool, help='Use external class', default=False)
    parser.add_argument('--return_embedding', type=bool, help='Return embedding', default=True)
    parser.add_argument('--is_data_splitted', type=bool, help='Is data splitted', default=False)
    parser.add_argument('--data_pickle_path', type=str, help='Data pickle path')
    parser.add_argument('--nn_train_loader_is_saved', type=bool, help='Is NN train loader saved', default=False)
    parser.add_argument('--nn_train_loader_path', type=str, help='NN train loader path')
    parser.add_argument('--do_save_nn_train_loader', type=bool, help='Do save NN train loader', default=True)
    parser.add_argument('--use_gpu', type=bool, help='Use GPU', default=True)
    parser.add_argument('--is_ssl', type=bool, help='Is SSL', default=False)
    parser.add_argument('--tensorboar_string', type=str, help='Tensorboard string')
    parser.add_argument('--generate_tensorboard_string_from_hyperparameters', type=bool, help='Generate tensorboard string from hyperparameters', default=True)
    parser.add_argument('--include_datetime', type=bool, help='Use experiment datetime as a suffix for tensorboard string', default=True)
    parser.add_argument('--resulting_dictionary_path', type=str, help='Resulting dictionary path')
    parser.add_argument('--verbose', type=bool, help='Verbose', default=False)

    args = parser.parse_args()

    # If a JSON config file is provided, override args with values from JSON
    if args.config:
        args = read_args_from_json(args.config)

    # Generate tensorboard string from hyperparameters
    if args.generate_tensorboard_string_from_hyperparameters:
        tensorboard_string = get_tensorboard_namer(experiment_name=args.experiment_name, is_binary_classification= args.is_binary_classification, 
                                                   is_hyper_sphere=args.is_hyper_sphere, hidden_sizes_scl=args.hidden_sizes_scl, embedding_size=args.embedding_size,
                                                   activation_scl=args.activation_scl, activation_clf=args.activation_clf,
                                                   train_batch_size=args.train_batch_size, lr_scl=args.lr_scl, lr_clf=args.lr_clf, lr_temporal=args.lr_temporal,
                                                   temperature_scl=args.temperature_scl, alpha_temporal=args.alpha_temporal, temporal_l2_reg_coeff=args.temporal_l2_reg_coeff,
                                                   do_nn_pairing=args.do_nn_pairing, nn_pairing_k=args.nn_pairing_k,
                                                   use_nn_clf=args.use_nn_clf, model_clf_str=args.model_clf_str, knn_clf=args.knn_clf, hidden_sizes_clf=args.hidden_sizes_clf, 
                                                   temporal_network=args.model_temporal_str, temporal_network_embedding_size=args.hidden_size_temporal, 
                                                   temporal_network_num_layers=args.num_temporal_layers,
                                                   tensorboard_suffix=args.tensorboar_string, include_datetime=args.include_datetime)
    else:
        tensorboard_string = args.tensorboard_string

    # Check if data is splitted
    if args.is_data_splitted:
        print(args.is_data_splitted)
        raise ValueError("Currently only unsplitted data is supported. Please set 'is_data_splitted' to 'False' and provide the path to the data pickle file with 'data_pickle_path' argument.")
    else:
        with open(args.data_pickle_path, "rb") as handle:
            data = pickle.load(handle)
        data_x, data_y = data["data_x"], data["data_y"]

    pretraining_dict, training_dict = run_temporal_scl(embedding_size=args.embedding_size, hidden_sizes_scl=args.hidden_sizes_scl, hidden_sizes_clf=args.hidden_sizes_clf,
                                                         hidden_size_temporal=args.hidden_size_temporal, num_temporal_layers=args.num_temporal_layers,
                                                         activation_scl=args.activation_scl, activation_clf=args.activation_clf, 
                                                         is_hyper_sphere=args.is_hyper_sphere, is_binary_classification=args.is_binary_classification,
                                                         train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size, test_batch_size=args.test_batch_size,
                                                         num_epochs_scl=args.num_epochs_scl, num_epochs_clf=args.num_epochs_clf, num_epochs_temporal=args.num_epochs_temporal,
                                                         num_epochs_scl_pretrain=args.num_epochs_scl_pretrain, num_epochs_clf_pretrain=args.num_epochs_clf_pretrain,
                                                         lr_scl=args.lr_scl, lr_clf=args.lr_clf, lr_temporal=args.lr_temporal,
                                                         temperature_scl=args.temperature_scl,
                                                         model_clf_str=args.model_clf_str, model_temporal_str=args.model_temporal_str,
                                                         criterion_temporal=args.criterion_temporal,
                                                         do_nn_pairing=args.do_nn_pairing, nn_pairing_k=args.nn_pairing_k,
                                                         alpha_temporal=args.alpha_temporal, temporal_l2_reg_coeff=args.temporal_l2_reg_coeff,
                                                         data_x=data_x, data_y=data_y, use_external_class=args.use_external_class, predict_raw_temporal=args.predict_raw_temporal,
                                                         valid_size=args.valid_size, test_size=args.test_size, seed=args.seed, shuffle_data=args.shuffle_data,
                                                         use_nn_clf=args.use_nn_clf, knn_clf=args.knn_clf, nn_metric=args.nn_metric, approximate_nn=args.approximate_nn,
                                                         return_embedding=args.return_embedding, saved_train_loader=args.nn_train_loader_is_saved, 
                                                         nn_train_loader_path=args.nn_train_loader_path,
                                                         do_save_train_loader=args.do_save_nn_train_loader, use_gpu=args.use_gpu,
                                                         tensorboar_string=tensorboard_string, experiment_name=args.experiment_name, verbose=args.verbose)
    
    pretraining_dict['args'] = args
    training_dict['args'] = args
    
    # Save results
    if args.resulting_dictionary_path:
        results_dict_path = args.resulting_dictionary_path
    else:
        results_dict_path = f"../results/{args.experiment_name}/result_dicts/"
    
    # check if results folders and the corresponding paths exist
    if not os.path.exists(results_dict_path):
        os.makedirs(results_dict_path)
    with open(f"{results_dict_path}{tensorboard_string}_pretraining_results.pkl", "wb") as handle:
        pickle.dump(pretraining_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{results_dict_path}{tensorboard_string}_training_results.pkl", "wb") as handle:
        pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to {results_dict_path}{tensorboard_string}_pretraining_results_.pkl and {results_dict_path}{tensorboard_string}_training_results.pkl")
    

if __name__ == '__main__':
    main()

