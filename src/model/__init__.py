from .dqn import DQNRecurrent, DQNFeedforward
from ..utils import bool_flag


models = {
    'dqn_ff': DQNFeedforward,
    'dqn_rnn': DQNRecurrent
}


def get_model_class(model_type):
    cls = models.get(model_type)
    if cls is None:
        raise RuntimeError(("unknown model type: '%s'. supported values "
                            "are: %s") % (model_type, ', '.join(models.keys())))
    return cls


def register_model_args(parser, args):
    """
    Parse model parameters.
    """
    # network type
    parser.add_argument("--network_type", type=str, default='dqn_ff',
                        help="Network type (dqn_ff / dqn_rnn)")
    parser.add_argument("--use_bn", type=bool_flag, default=False,
                        help="Use batch normalization in CNN network")

    # model parameters
    params, _ = parser.parse_known_args(args)
    network_class = get_model_class(params.network_type)
    network_class.register_args(parser)
    network_class.validate_params(parser.parse_known_args(args)[0])

    # parameters common to all models
    parser.add_argument("--clip_delta", type=float, default=1.0,
                        help="Clip delta")
    parser.add_argument("--variable_dim", type=str, default='32',
                        help="Game variables embeddings dimension")
    parser.add_argument("--bucket_size", type=str, default='1',
                        help="Bucket size for game variables")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer dimension")
    parser.add_argument("--update_frequency", type=int, default=4,
                        help="Update frequency (1 for every time)")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="Dropout")
    parser.add_argument("--optimizer", type=str, default="rmsprop,lr=0.0002",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")

    # check common parameters
    params, _ = parser.parse_known_args(args)
    assert params.clip_delta >= 0
    assert params.update_frequency >= 1
    assert 0 <= params.dropout < 1
