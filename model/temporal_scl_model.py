from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

############################################################
################## Models Definition #######################
############################################################

class TSCL_Encoder_Network(nn.Module):
    """Encoder Network for Temporal-SCL"""
    def __init__(self, input_size, hidden_sizes=[128], embedding_size=128, activation="ReLU", is_hyper_sphere=True):

        super(TSCL_Encoder_Network, self).__init__()
        self.is_hyper_sphere = is_hyper_sphere
        
        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Sigmoid": nn.Sigmoid()
        }

        num_layers = len(hidden_sizes) + 1

        # Checking input parameters        
        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2.")

        self.activation = activations.get(activation)
        if not self.activation:
            raise ValueError(f"Activation of {activation} is not acceptable.")
        
        # Last layer is the embedding layer
        if len(hidden_sizes) != num_layers - 1:
            raise ValueError("Number of hidden layers must be less than or equal to number of layers - 1.")
        
        # Create layers dynamically
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [embedding_size]

        # Add layers to the network
        for i in range(num_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Do not add activation after the last layer
            if i < num_layers - 1:
                layers.append(self.activation)
        
        # Create the sequential container
        self.embedding_front = nn.Sequential(*layers)
    
    def forward(self, x):
        embedding_out = self.embedding_front(x)
        if self.is_hyper_sphere:
            embedding_out = F.normalize(embedding_out, dim=1)
        return embedding_out


class TSL_Predictor_Network(nn.Module):
    """Predictor Network for Temporal-SCL"""
    def __init__(self, embedding_dim, num_classes, hidden_sizes=[], activation="ReLU"):
        super(TSL_Predictor_Network, self).__init__()

        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Sigmoid": nn.Sigmoid()
        }

        num_layers = len(hidden_sizes) + 1

        # Checking input parameters
        self.activation = activations.get(activation)
        if not self.activation:
            raise ValueError(f"Activation of {activation} is not acceptable.")
        
        # Define the layers based on num_layers and hidden_sizes
        layer_sizes = [embedding_dim] + hidden_sizes + [num_classes]
        
        # The number of hidden layers should be one less than the number of layers
        if len(hidden_sizes) != num_layers - 1:
            raise ValueError("Number of hidden layers should be equal to num_layers - 1.")

        # Constructing layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add activation if it is not the last layer
            if i < num_layers - 1:
                layers.append(self.activation)
        
        # Assign the layers to the module
        self.predictive_head = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictive_head(x)
    

class TSCL_Temporal_Network(nn.Module):
    '''Temporal Regularizer Network for Temporal-SCL'''
    def __init__(self, n_features, dimension=128, num_rnn_layers=1, rnn_type='LSTM', output_dim=None):
        super(TSCL_Temporal_Network, self).__init__()
        
        self.n_features = n_features
        self.dimension = dimension
        self.rnn_type = rnn_type

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.n_features,
                               hidden_size=self.dimension,
                               num_layers=num_rnn_layers,
                               batch_first=True)
            # self.drop = nn.Dropout(p=0.5)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.n_features,
                              hidden_size=self.dimension,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        else:
            raise ValueError(f"RNN type of {rnn_type} is not acceptable.")
        
        self.output_dim = output_dim if output_dim is not None else self.n_features
        self.fc = nn.Linear(self.dimension, self.output_dim)

    def forward(self, seq, seq_len):
        ''' This function handles variable length input sequence.'''
        packed_input = pack_padded_sequence(seq, seq_len.to('cpu'), batch_first=True)
        packed_output, _ = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), seq_len - 1, :self.dimension]
        seq_fea = self.fc(out_forward)
        return seq_fea
