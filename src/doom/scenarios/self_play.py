import importlib
import argparse
import torch.multiprocessing as mp

# Arnold
from ...parameter_server import ParameterServer


def worker_fn_factory(main):
    def worker_fn(rank, parser, args, param_server):
        param_server.set_rank(rank)
        main(parser, args, parameter_server=param_server)
    return worker_fn


def main(_parser, args):
    parser = argparse.ArgumentParser(description='Arnold runner')
    parser.add_argument("--execute", default="deathmatch_rockets",
                        help="Script to run")
    parser.add_argument("--num_players", type=int, default=2,
                        help="Number of agents to run")
    parser.add_argument("--num_games", type=int, default=1,
                        help="Number of games to run")
    params, remaining_args = parser.parse_known_args(args)
    module = importlib.import_module('...scenarios.' + params.execute,
                                     package=__name__)
    assert params.num_games % params.num_games == 0
    assert '--player_rank' not in remaining_args
    assert '--players_per_game' not in remaining_args
    players_per_game = params.num_players // params.num_games
    assert players_per_game in range(1, 9)
    processes = []
    param_server = ParameterServer(params.num_players)
    for i in range(params.num_players):
        subprocess_args = ['--players_per_game', str(players_per_game),
                           '--player_rank', str(i)]
        subprocess_args += remaining_args
        proc = mp.Process(target=worker_fn_factory(module.main),
                          args=(i, _parser, subprocess_args, param_server))
        proc.start()
        processes.append(proc)
    for p in processes:
        p.join()
