import os
import json
import torch
import pickle
import numpy as np
from logging import getLogger


# Arnold
from ...utils import set_num_threads, get_device_mapping
from ...model import register_model_args, get_model_class
from ...trainer import ReplayMemoryTrainer
from ...args import finalize_args
from ..game_features import GameFeaturesConfusionMatrix
from ..game import Game
from ..actions import ActionBuilder


logger = getLogger()


def main(parser, args, parameter_server=None):

    # register model parameters / parse parameters
    register_model_args(parser, args)
    params = parser.parse_args(args)

    # Game variables / Game features / feature maps
    params.game_variables = [('health', 101), ('sel_ammo', 301)]
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
    assert params.action_combinations in ['turn_lr+attack', 'attack+turn_lr',
                                          'turn_lr;attack', 'attack;turn_lr']
    action_builder = ActionBuilder(params)

    # Initialize the game
    game = Game(
        scenario='defend_the_center',
        action_builder=action_builder,
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
        render_crosshair=params.render_crosshair,
        render_weapon=params.render_weapon,
        freelook=params.freelook,
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
        evaluate_defend_the_center(game, network, params)
    else:
        logger.info('Starting experiment...')
        if params.network_type.startswith('dqn'):
            trainer_class = ReplayMemoryTrainer
        else:
            raise RuntimeError("unknown network type " + params.network_type)
        trainer_class(params, game, network, evaluate_defend_the_center,
                      parameter_server=parameter_server).run()


def evaluate_defend_the_center(game, network, params, n_train_iter=None):
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
    n_kills = []

    while True:
        n_iter += 1

        if game.is_player_dead() or game.is_episode_finished():
            # store the number of kills
            n_kills.append(game.properties['score'])
            logger.info("%i kills." % game.properties['score'])
            logger.info("===============")
            if len(n_kills) == params.eval_episodes:
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
    logger.info("%i iterations on %i episodes." % (n_iter, len(n_kills)))
    if n_features != 0:
        gf_confusion.print_statistics()
    logger.info("Kills by episode: %s" % str(n_kills))
    logger.info("%f kills / episode average." % np.mean(n_kills))
    to_log = {'min_n_kills': float(np.min(n_kills)),
              'max_n_kills': float(np.max(n_kills)),
              'mean_n_kills': float(np.mean(n_kills))}
    if n_train_iter is not None:
        to_log['n_iter'] = n_train_iter
    logger.info("__log__:%s" % json.dumps(to_log))

    # evaluation score
    return np.mean(n_kills)
