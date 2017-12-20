import torch
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
from .bucketed_embedding import BucketedEmbedding


logger = getLogger()


def get_recurrent_module(module_type):
    if module_type == 'rnn':
        return nn.RNN
    elif module_type == 'gru':
        return nn.GRU
    elif module_type == 'lstm':
        return nn.LSTM
    else:
        raise Exception("Unknown recurrent module type: '%s'" % module_type)


def value_loss(delta):
    """
    MSE Loss / Smooth L1 Loss / Huber Loss
    """
    assert delta >= 0
    if delta == 0:
        # MSE Loss
        return nn.MSELoss()
    elif delta == 1:
        # Smooth L1 Loss
        return nn.SmoothL1Loss()
    else:
        # Huber Loss
        def loss_fn(input, target):
            diff = (input - target).abs()
            diff_delta = diff.cmin(delta)
            loss = diff_delta * (diff - diff_delta / 2)
            return loss.mean()
        return loss_fn


def build_CNN_network(module, params):
    """
    Build CNN network.
    """
    # model parameters
    module.hidden_dim = params.hidden_dim
    module.dropout = params.dropout
    module.n_actions = params.n_actions

    # screen input format - for RNN, we only take one frame at each time step
    if hasattr(params, 'recurrence') and params.recurrence != '':
        in_channels = params.n_fm
    else:
        in_channels = params.n_fm * params.hist_size
    height = params.height
    width = params.width
    logger.info('Input shape: %s' % str((params.n_fm, height, width)))

    # convolutional layers
    module.conv = nn.Sequential(*filter(bool, [
        nn.Conv2d(in_channels, 32, (8, 8), stride=(4, 4)),
        None if not params.use_bn else nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
        None if not params.use_bn else nn.BatchNorm2d(64),
        nn.ReLU(),

        # nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
        # None if not params.use_bn else nn.BatchNorm2d(64),
        # nn.ReLU(),

        # None if module.dropout == 0 else nn.Dropout(module.dropout)
    ]))

    # get the size of the convolution network output
    x = Variable(torch.FloatTensor(1, in_channels, height, width).zero_())
    module.conv_output_dim = module.conv(x).nelement()


def build_game_variables_network(module, params):
    """
    Build game variables network (health, ammo, etc.)
    """
    module.game_variables = params.game_variables
    module.n_variables = params.n_variables
    module.game_variable_embeddings = []
    for i, (name, n_values) in enumerate(params.game_variables):
        embeddings = BucketedEmbedding(params.bucket_size[i], n_values,
                                       params.variable_dim[i])
        setattr(module, '%s_emb' % name, embeddings)
        module.game_variable_embeddings.append(embeddings)


def build_game_features_network(module, params):
    """
    Build game features network.
    """
    module.game_features = params.game_features
    if module.game_features:
        module.n_features = module.game_features.count(',') + 1
        module.proj_game_features = nn.Sequential(
            nn.Dropout(module.dropout),
            nn.Linear(module.conv_output_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(module.dropout),
            nn.Linear(params.hidden_dim, module.n_features),
            nn.Sigmoid()
        )
    else:
        module.n_features = 0
