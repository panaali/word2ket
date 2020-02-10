# Attentional Seq2seq model.
# Hyperparameters not specified here will take the default values.

num_units = 8000
beam_width = 5
dropout = 0.2

embedder = {
    'dim': num_units
}

encoder = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': num_units
        },
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    }
}

decoder = {
    'rnn_cell': {
        'kwargs': {
            'num_units': num_units
        },
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    },
    'attention': {
        'kwargs': {
            'num_units': num_units,
        },
        'attention_layer_size': num_units
    },
    'max_decoding_length_infer': 60,
}

opt = {
    'optimizer': {
        'type':  'Adam',
        'kwargs': {
            'lr': 0.001,
        },
    },
}
