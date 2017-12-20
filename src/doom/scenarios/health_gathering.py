import os
import json
import torch
import pickle
import numpy as np
from logging import getLogger


# Arnold
from ...utils import set_num_threads, get_device_mapping, bool_flag
from ...model import register_model_args, get_model_class
from ...trainer import ReplayMemoryTrainer
from ...args import finalize_args
from ..game_features import GameFeaturesConfusionMatrix
from ..game import Game
from ..actions import ActionBuilder


logger = getLogger()


def register_scenario_args(parser):
    # supreme mode for health gathering
    parser.add_argument("--supreme", type=bool_flag, default=False,
                        help="Use the supreme mode for health gathering")


def main(parser, args, parameter_server=None):

    # register model and scenario parameters / parse parameters
    register_model_args(parser, args)
    register_scenario_args(parser)
    params = parser.parse_args(args)

    # Game variables / Game features / feature maps
    params.game_variables = [('health', 101)]
    finalize_args(params)

    # Training / Evaluation settings
    params.episode_time = 120  # episode maximum duration (in seconds)
    params.eval_freq = 30000   # time (in iterations) between 2 evaluations
    params.eval_episodes = 20  # number of episodes for evaluation

    # log experiment parameters
    with open(os.path.join(params.dump_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in dict(vars(params)).items()))

    # use only 1 CPU thread / set GPU ID if required
    set_num_threads(1)
    if params.gpu_id >= 0:
        torch.cuda.set_device(params.gpu_id)

    # Action builder
    assert 'attack' not in params.action_combinations
    action_builder = ActionBuilder(params)

    # Give a reward for survival
    reward_values = {'BASE_REWARD': 0.01, 'MEDIKIT': 1, 'INJURED': 0}

    # Initialize the game
    game = Game(
        scenario='health_gathering%s' % ('_supreme' if params.supreme else ''),
        action_builder=action_builder,
        reward_values=reward_values,
        score_variable='USER1',
        freedoom=params.freedoom,
        use_screen_buffer=params.use_screen_buffer,
        use_depth_buffer=params.use_depth_buffer,
        labels_mapping=params.labels_mapping,
        game_features=params.game_features,
        mode='PLAYER',
        player_rank=params.player_rank,
        players_per_game=params.players_per_game,
        render_hud=params.render_hud,
        render_crosshair=False,
        render_weapon=False,
        freelook=params.freelook,
        respawn_protect=False,
        visible=params.visualize
    )

    # Network initialization and optional reloading
    network = get_model_class(params.network_type)(params)
    if params.reload:
        logger.info('Reloading model from %s...' % params.reload)
        model_path = os.path.join(params.dump_path, params.reload)
        map_location = get_device_mapping(params.gpu_id)
        reloaded = torch.load(model_path, map_location=map_location)
        network.module.load_state_dict(reloaded)
    assert params.n_features == network.module.n_features

    # Parameter server (multi-agent training, self-play, etc.)
    if parameter_server:
        assert params.gpu_id == -1
        parameter_server.register_model(network.module)

    # Visualize only
    if params.evaluate:
        evaluate_health_gathering(game, network, params)
    else:
        logger.info('Starting experiment...')
        if params.network_type.startswith('dqn'):
            trainer_class = ReplayMemoryTrainer
        else:
            raise RuntimeError("unknown network type " + params.network_type)
        trainer_class(params, game, network, evaluate_health_gathering,
                      parameter_server=parameter_server).run()


def evaluate_health_gathering(game, network, params, n_train_iter=None):
    """
    Evaluate the model.
    """
    logger.info('Evaluating the model...')
    map_id = params.map_ids_test[0]
    game.start(map_id=map_id, episode_time=params.episode_time, log_events=True)
    network.reset()
    network.module.eval()
    n_features = params.n_features

    n_iter = 0
    last_states = []
    if n_features > 0:
        gf_confusion = GameFeaturesConfusionMatrix([map_id], n_features)
    survival_time = []

    while True:
        n_iter += 1

        if game.is_player_dead() or game.is_episode_finished():
            # store the number of kills
            survival_time.append(game.game.get_episode_time())
            logger.info("Survived for %i steps." % game.game.get_episode_time())
            logger.info("===============")
            if len(survival_time) == params.eval_episodes:
                break
            game.new_episode()
            network.reset()

        # observe the game state / select the next action
        game.observe_state(params, last_states)
        action = network.next_action(last_states)
        pred_features = network.pred_features

        # game features
        assert (pred_features is None) ^ n_features
        if n_features:
            assert pred_features.nelement() == params.n_features
            pred_features = pred_features.data.cpu().numpy().ravel()
            gf_confusion.update_predictions(pred_features,
                                            last_states[-1].features,
                                            map_id)

        sleep = 0.001 if params.evaluate else None
        game.make_action(action, params.frame_skip, sleep=sleep)

    # close the game
    game.close()

    # log the number of iterations and statistics
    logger.info("%i iterations on %i episodes." % (n_iter, len(survival_time)))
    if n_features != 0:
        gf_confusion.print_statistics()
    logger.info("Survival time by episode: %s" % str(survival_time))
    logger.info("%f survival time / episode average." % np.mean(survival_time))
    to_log = {'min_survival_time': float(np.min(survival_time)),
              'max_survival_time': float(np.max(survival_time)),
              'mean_survival_time': float(np.mean(survival_time))}
    if n_train_iter is not None:
        to_log['n_iter'] = n_train_iter
    logger.info("__log__:%s" % json.dumps(to_log))

    # evaluation score
    return np.mean(survival_time)
