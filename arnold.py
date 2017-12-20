import argparse
import os
import vizdoom
from src.utils import get_dump_path
from src.logger import get_logger
from src.args import parse_game_args


parser = argparse.ArgumentParser(description='Arnold runner')
parser.add_argument("--main_dump_path", type=str, default="./dumped",
                    help="Main dump path")
parser.add_argument("--exp_name", type=str, default="default",
                    help="Experiment name")
args, remaining = parser.parse_known_args()
assert len(args.exp_name.strip()) > 0

# create a directory for the experiment / create a logger
dump_path = get_dump_path(args.main_dump_path, args.exp_name)
logger = get_logger(filepath=os.path.join(dump_path, 'train.log'))
logger.info('========== Running DOOM ==========')
logger.info('Experiment will be saved in: %s' % dump_path)

# load DOOM
parse_game_args(remaining + ['--dump_path', dump_path])
