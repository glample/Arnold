import os
import json
import torch
import pickle
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
    """
    Register scenario parameters.
    """
    parser.add_argument("--wad", type=str, default="",
                        help="WAD scenario filename")
    parser.add_argument("--n_bots", type=int, default=8,
                        help="Number of ACS bots in the game")
    parser.add_argument("--reward_values", type=str, default="",
                        help="reward_values")
    parser.add_argument("--randomize_textures", type=bool_flag, default=False,
                        help="Randomize textures during training")
    parser.add_argument("--init_bots_health", type=int, default=100,
                        help="Initial bots health during training")


def parse_reward_values(reward_values):
    """
    Parse rewards values.
    """
    values = reward_values.split(',')
    reward_values = {}
    for x in values:
        if x == '':
            continue
        split = x.split('=')
        assert len(split) == 2
        reward_values[split[0]] = float(split[1])
    return reward_values


def main(parser, args, parameter_server=None):
    """
    Deathmatch running script.
    """
    # register model and scenario parameters / parse parameters
    register_model_args(parser, args)
    register_scenario_args(parser)
    params = parser.parse_args(args)
    params.human_player = params.human_player and params.player_rank == 0

    # Game variables / Game features / feature maps
    params.game_variables = [('health', 101), ('sel_ammo', 301)]
    finalize_args(params)

    # Training / Evaluation parameters
    params.episode_time = None  # episode maximum duration (in seconds)
    params.eval_freq = 20000    # time (in iterations) between 2 evaluations
    params.eval_time = 900      # evaluation time (in seconds)

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
    action_builder = ActionBuilder(params)

    # Initialize the game
    game = Game(
        scenario=params.wad,
        action_builder=action_builder,
        reward_values=parse_reward_values(params.reward_values),
        score_variable='USER2',
        freedoom=params.freedoom,
        # screen_resolution='RES_400X225',
        use_screen_buffer=params.use_screen_buffer,
        use_depth_buffer=params.use_depth_buffer,
        labels_mapping=params.labels_mapping,
        game_features=params.game_features,
        mode=('SPECTATOR' if params.human_player else 'PLAYER'),
        player_rank=params.player_rank,
        players_per_game=params.players_per_game,
        render_hud=params.render_hud,
        render_crosshair=params.render_crosshair,
        render_weapon=params.render_weapon,
        freelook=params.freelook,
        visible=params.visualize,
        n_bots=params.n_bots,
        use_scripted_marines=True
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
        evaluate_deathmatch(game, network, params)
    else:
        logger.info('Starting experiment...')
        if params.network_type.startswith('dqn'):
            trainer_class = ReplayMemoryTrainer
        else:
            raise RuntimeError("unknown network type " + params.network_type)
        trainer_class(params, game, network, evaluate_deathmatch,
                      parameter_server=parameter_server).run()


def evaluate_deathmatch(game, network, params, n_train_iter=None):
    """
    Evaluate the model.
    """
    logger.info('Evaluating the model...')
    game.statistics = {}

    n_features = params.n_features
    if n_features > 0:
        confusion = GameFeaturesConfusionMatrix(params.map_ids_test, n_features)

    # evaluate on every test map
    for map_id in params.map_ids_test:

        logger.info("Evaluating on map %i ..." % map_id)
        game.start(map_id=map_id, log_events=True,
                   manual_control=(params.manual_control and not params.human_player))
        game.randomize_textures(False)
        game.init_bots_health(100)
        network.reset()
        network.module.eval()

        n_iter = 0
        last_states = []

        while n_iter * params.frame_skip < params.eval_time * 35:
            n_iter += 1

            if game.is_player_dead():
                game.respawn_player()
                network.reset()

            while game.is_player_dead():
                logger.warning('Player %i is still dead after respawn.' %
                               params.player_rank)
                game.respawn_player()

            # observe the game state / select the next action
            game.observe_state(params, last_states)
            action = network.next_action(last_states).tolist()
            pred_features = network.pred_features

            # game features
            assert (pred_features is None) ^ n_features
            if n_features:
                assert pred_features.size() == (params.n_features,)
                pred_features = pred_features.data.cpu().numpy().ravel()
                confusion.update_predictions(pred_features,
                                             last_states[-1].features,
                                             game.map_id)

            sleep = 0.01 if params.evaluate else None
            game.make_action(action, params.frame_skip, sleep=sleep)

        # close the game
        game.close()

    # log the number of iterations and statistics
    logger.info("%i iterations" % n_iter)
    if n_features != 0:
        confusion.print_statistics()
    game.print_statistics()
    to_log = ['kills', 'deaths', 'suicides', 'frags', 'k/d']
    to_log = {k: game.statistics['all'][k] for k in to_log}
    if n_train_iter is not None:
        to_log['n_iter'] = n_train_iter
    logger.info("__log__:%s" % json.dumps(to_log))

    # evaluation score
    return game.statistics['all']['frags']
