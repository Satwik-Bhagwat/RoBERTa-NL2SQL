import torch
import os
from seq2sql_model_classes import Seq2SQL_v1
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig

device = torch.device("cuda")

def get_electra_model():

    # Initializing a XLNet configuration
    configuration = ElectraConfig()

    # Initializing a model from the configuration
    electra_Model = ElectraModel(configuration).from_pretrained("google/electra-small-discriminator")
    electra_Model.to(device)

    # Accessing the model configuration
    configuration = electra_Model.config

    #get the XLNet Tokenizer
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

    return electra_Model, tokenizer, configuration


def get_seq2sql_model(Xlnet_hidden_layer_size, number_of_layers = 2,
                    hidden_vector_dimensions = 100,
                    number_lstm_layers = 2,
                    dropout_rate = 0.3,
                    load_pretrained_model=False, model_path=None):
    
    '''
    
    get_seq2sql_model
    Arguments:
    Xlnet_hidden_layer_size: sizes of hidden layers of Xlnet model
    number_of_layers : total number of layers
    hidden_vector_dimensions : dimensions of hidden vectors
    number_lstm_layers : total number of lstm layers
    dropout_rate : value of dropout rate
    load_pretrained_model : want to load pretrained model(true or false)
    model_path : The path to the directory in which the model is contained
    
    Returns:
    model: returns the model
    
    '''

    # number_of_layers = "The Number of final layers of Xlnet to be used in downstream task."
    # hidden_vector_dimensions : "The dimension of hidden vector in the seq-to-SQL module."
    # number_lstm_layers : "The number of LSTM layers." in seqtosqlmodule

    sql_main_operators = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    sql_conditional_operators = ['=', '>', '<', 'OP']

    number_of_neurons = Xlnet_hidden_layer_size * number_of_layers  # Seq-to-SQL input vector dimenstion

    model = Seq2SQL_v1(number_of_neurons, hidden_vector_dimensions, number_lstm_layers, dropout_rate, len(sql_conditional_operators), len(sql_main_operators))
    model = model.to(device)

    if load_pretrained_model:
        assert model_path != None
        if torch.cuda.is_available():
            res = torch.load(model_path)
        else:
            res = torch.load(model_path, map_location='cpu')
        model.load_state_dict(res['model'])

    return model

def get_optimizers(model, model_Xlnet,learning_rate_model=1e-3,learning_rate_Xlnet=1e-5):
    '''
    get_optimizers
    Arguments:
    model: returned model from get_seq2sql_model
    model_Xlnet : returned model from get_Xlnet_model
    fine_tune : want to fine tune(true or false)
    learning_rate_model : learning rate of model (from get_seq2sql_model)
    learning_rate_Xlnet : learning rate of Xlnet model (from get_Xlnet_model)
    
    Returns:
    opt: returns the optimised model (from get_seq2sql_model)
    opt_Xlnet : returns the optimised Xlnet model (from get_Xlnet_model)
    '''

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate_model, weight_decay=0)

    opt_Xlnet = torch.optim.Adam(filter(lambda p: p.requires_grad, model_Xlnet.parameters()),
                                lr=learning_rate_Xlnet, weight_decay=0)

    return opt, opt_Xlnet
