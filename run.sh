#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -a|--n_agents)
    A="$2"
    shift
    shift
    ;;
    -b|--n_bots)
    B="$2"
    shift
    shift
    ;;
    -m|--map_id)
    M="$2"
    shift
    shift
    ;;
    -h|--human_player)
    H="$2"
    shift
    shift
    ;;
    --default)
    DEFAULT=YES
    shift
    ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$A" == "" ]; then
	A=1
fi
if [ "$B" == "" ]; then
	B=2
fi
if [ "$M" == "" ]; then
	M=2
fi
if [ "$H" == "" ]; then
	H=0
fi

P=$((A+H))

if [ "$1" == "defend_the_center" ]; then
python arnold.py --exp_name test --main_dump_path $PWD/dumped \
--scenario defend_the_center --frame_skip 2 --action_combinations "turn_lr+attack" \
--reload $PWD/pretrained/defend_the_center.pth --evaluate 1 --visualize 1 --gpu_id -1
fi

if [ "$1" == "health_gathering" ]; then
python arnold.py --exp_name test --main_dump_path $PWD/dumped \
--scenario health_gathering --supreme 1 \
--frame_skip 4 --action_combinations "move_fb;turn_lr" \
--reload $PWD/pretrained/health_gathering.pth --evaluate 1 --visualize 1 --gpu_id -1
fi

if [ "$1" == "track1" ]; then
echo "Number of agents: ${A}"
echo "Number of bots  : ${B}"
echo "Map ID          : 1"
echo "Human player    : ${H}"
python arnold.py --exp_name test --main_dump_path $PWD/dumped \
--frame_skip 3 --action_combinations "attack+move_lr;turn_lr;move_fb" \
--network_type "dqn_rnn" --recurrence "lstm" --n_rec_layers 1 --hist_size 4 --remember 1 \
--labels_mapping "" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 \
--speed "on" --crouch "off" --map_ids_test 1 --manual_control 1 \
--scenario "self_play" --execute "deathmatch" --wad "deathmatch_rockets" \
--num_players ${P} --n_bots ${B} --human_player ${H} \
--reload $PWD/pretrained/vizdoom_2017_track1.pth --evaluate 1 --visualize 1 --gpu_id -1
fi

if [ "$1" == "track2" ]; then
echo "Number of agents: ${A}"
echo "Number of bots  : ${B}"
echo "Map ID          : ${M}"
echo "Human player    : ${H}"
python arnold.py --exp_name test --main_dump_path $PWD/dumped \
--frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" \
--network_type "dqn_rnn" --recurrence "lstm" --n_rec_layers 1 --hist_size 4 --remember 1 \
--labels_mapping "" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 \
--speed "on" --crouch "off" --map_ids_test ${M} --manual_control 1 \
--scenario "self_play" --execute "deathmatch" --wad "full_deathmatch" \
--num_players ${P} --n_bots ${B} --human_player ${H} \
--reload $PWD/pretrained/vizdoom_2017_track2.pth --evaluate 1 --visualize 1 --gpu_id -1
 fi

if [ "$1" == "shotgun" ]; then
echo "Number of agents: ${A}"
echo "Number of bots  : ${B}"
echo "Map ID          : 7"
echo "Human player    : ${H}"
python arnold.py --exp_name test --main_dump_path . \
--frame_skip 3 --action_combinations "move_fb+move_lr;turn_lr;attack" \
--network_type "dqn_rnn" --recurrence "lstm" --n_rec_layers 1 --hist_size 6 --remember 1 \
--labels_mapping "0" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 \
--speed "on" --crouch "off" --map_ids_test 7 --manual_control 1 \
--scenario "self_play" --execute "deathmatch" --wad "deathmatch_shotgun" \
--num_players ${P} --n_bots ${B} --human_player ${H} \
--reload $PWD/pretrained/deathmatch_shotgun.pth --evaluate 1 --visualize 1 --gpu_id -1
fi
