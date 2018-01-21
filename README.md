# Arnold

Arnold is a PyTorch implementation of the agent presented in *Playing FPS Games with Deep Reinforcement Learning* (https://arxiv.org/abs/1609.05521), and that won the 2017 edition of the [*ViZDoom AI Competition*](http://vizdoom.cs.put.edu.pl/competition-cig-2017).

![example](./docs/example.gif) 

### This repository contains:
- The source code to train DOOM agents
- A package with 17 selected maps that can be used for training and evaluation
- 5 pretrained models that you can visualize and play against, including the ones that won the ViZDoom competition


## Installation

#### Dependencies
Arnold was tested successfully on Mac OS and Linux distributions. You will need:
- Python 2/3 with NumPy and OpenCV
- PyTorch
- ViZDoom

Follow the instructions on https://github.com/mwydmuch/ViZDoom to install ViZDoom. Be sure that you can run `import vizdoom` in Python from any directory. To do so, you can either install the library with `pip`, or compile it, then move it to the `site-packages` directory of your Python installation, as explained here: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Quickstart.md.


## Code structure

    .
    ├── pretrained                    # Examples of pretrained models
    ├── resources
        ├── freedoom2.wad             # DOOM resources file (containing all textures)
        └── scenarios                 # Folder containing all scenarios
            ├── full_deathmatch.wad   # Scenario containing all deathmatch maps
            ├── health_gathering.wad  # Simple test scenario
            └── ...
    ├── src                           # Source files
        ├── doom                      # Game interaction / API / scenarios
        ├── model                     # DQN / DRQN implementations
        └── trainer                   # Folder containing training scripts
    ├── arnold.py                     # Main file
    └── README.md


## Scenarios / Maps

## Train a model

There are many parameters you can tune to train a model.


```bash
python arnold.py

## General parameters about the game
--freedoom "true"                # use freedoom resources
--height 60                      # screen height
--width 108                      # screen width
--gray "false"                   # use grayscale screen
--use_screen_buffer "true"       # use the screen buffer (what the player sees)
--use_depth_buffer "false"       # use the depth buffer
--labels_mapping ""              # use extra feature maps for specific objects
--game_features "target,enemy"   # game features prediction (auxiliary tasks)
--render_hud "false"             # render the HUD (status bar in the bottom of the screen)
--render_crosshair "true"        # render crosshair (targeting aid in the center of the screen)
--render_weapon "true"           # render weapon
--hist_size 4                    # history size
--frame_skip 4                   # frame skip (1 = keep every frame)

## Agent allowed actions
--action_combinations "attack+move_lr;turn_lr;move_fb"  # agent allowed actions
--freelook "false"               # allow the agent to look up and down
--speed "on"                     # make the agent run
--crouch "off"                   # make the agent crouch

## Training parameters
--batch_size 32                  # batch size
--replay_memory_size 1000000     # maximum number of frames in the replay memory
--start_decay 0                  # epsilon decay iteration start
--stop_decay 1000000             # epsilon decay iteration end
--final_decay 0.1                # final epsilon value
--gamma 0.99                     # discount factor gamma
--dueling_network "false"        # use a dueling architecture
--clip_delta 1.0                 # clip the delta loss
--update_frequency 4             # DQN update frequency
--dropout 0.5                    # dropout on CNN output layer
--optimizer "rmsprop,lr=0.0002"  # network optimizer

## Network architecture
--network_type "dqn_rnn"         # network type (dqn_ff / dqn_rnn)
--recurrence "lstm"              # recurrent network type (rnn / gru / lstm)
--n_rec_layers 1                 # number of layers in the recurrent network
--n_rec_updates 5                # number of updates by sample
--remember 1                     # remember all frames during evaluation
--use_bn "off"                   # use BatchNorm when processing the screen
--variable_dim "32"              # game variables embeddings dimension
--bucket_size "[10, 1]"          # bucket game variables (typically health / ammo)
--hidden_dim 512                 # hidden layers dimension

## Scenario parameters (these parameters will differ based on the scenario)
--scenario "deathmatch"          # scenario
--wad "full_deathmatch"          # WAD file (scenario file)
--map_ids_train "2,3,4,5"        # maps to train the model
--map_ids_test "6,7,8"           # maps to test the model
--n_bots 8                       # number of enemy bots
--randomize_textures "true"      # randomize walls / floors / ceils textures during training
--init_bots_health 20            # reduce initial life of enemy bots (helps a lot when using pistol)

## Various
--exp_name new_train             # experiment name
--dump_freq 200000               # periodically dump the model
--gpu_id -1                      # GPU ID (-1 to run on CPU)
```

Once your agent is trained, you can visualize it by running the same command, and using the following extra arguments:
```bash
--visualize 1                    # visualize the model (render the screen)
--evaluate 1                     # evaluate the agent
--manual_control 1               # manually make the agent turn about when it gets stuck
--reload PATH                    # path where the trained agent was saved
```


Here are some examples of training commands for 3 different scenarios:

#### Defend the center

In this scenario the agent is in the middle of a circular map. Monsters are regularly appearing on the sides and are walking towards the agent. The agent is given a pistol and limited ammo, and must turn around and kill the monsters before they reach it. The following command trains a standard DQN, that should reach the optimal performance of 56 frags (the number of bullets in the pistol) in about 4 million steps:

```bash
python arnold.py --scenario defend_the_center --action_combinations "turn_lr+attack" --frame_skip 2
```

#### Health gathering

In this scenario the agent is walking on lava, and is losing health points at each time step. The agent has to move and collect as many health pack as possible in order to survive. The objective is to survive the longest possible time.

```bash
python arnold.py --scenario health_gathering --action_combinations "move_fb;turn_lr" --frame_skip 5
```

This scenario is very easy and the model quickly reaches the maximum survival time of 2 minutes (35 * 120 = 4200 frames). The scenario also provides a `supreme` mode, in which the map is more complicated and where the health packs are much harder to collect:

```bash
python arnold.py --scenario health_gathering --action_combinations "move_fb;turn_lr" --frame_skip 5 --supreme 1
```

In this scenario, the agent takes about 1.5 million steps to reach the maximum survival time (but often dies before the end).

#### Deathmatch

In this scenario, the agent is trained to fight against the built-in bots of the game. Here is a command to train the agent using game features prediction (as described in [1]), and a DRQN:

```bash
python arnold.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 \
--action_combinations "move_fb;move_lr;turn_lr;attack" --frame_skip 4 \
--game_features "enemy" --network_type dqn_rnn --recurrence lstm --n_rec_updates 5
```


## Pretrained models

#### Defend the center / Health gathering

We provide a pretrained model for each of these scenarios. You can visualize them by running:

```bash
./run.sh defend_the_center
```

or

```bash
./run.sh health_gathering
```

#### Visual Doom AI Competition 2017

We release the two agents submitted to the first and second tracks of the ViZDoom AI 2017 Competition. You can visualize them playing against the built-in bots using the following commands:

##### Track 1 - Arnold vs 10 built-in AI bots
```bash
./run.sh track1 --n_bots 10
```

##### Track 2 - Arnold vs 10 built-in AI bots - Map 2
```bash
./run.sh track2 --n_bots 10 --map_id 2
```

##### Track 2 - 4 Arnold playing against each other - Map 3
```bash
./run.sh track2 --n_bots 0 --map_id 3 --n_agents 4
```

We also trained an agent on a single map, using a same weapon (the SuperShotgun). This agent is extremely difficult to beat.

##### Shotgun - 4 Arnold playing against each other
```bash
./run.sh shotgun --n_bots 0 --n_agents 4
```

##### Shotgun - 3 Arnold playing against each other + 1 human player (to play against the agent)
```bash
./run.sh shotgun --n_bots 0 --n_agents 3 --human_player 1
```


## References

If you found this code useful, please consider citing:

[1] G. Lample\* and D.S. Chaplot\*, [*Playing FPS Games with Deep Reinforcement Learning*](https://arxiv.org/abs/1609.05521)
```
@inproceedings{lample2017playing,
  title={Playing FPS Games with Deep Reinforcement Learning.},
  author={Lample, Guillaume and Chaplot, Devendra Singh},
  booktitle={Proceedings of AAAI},
  year={2017}
}
```


[2] D.S. Chaplot\* and G. Lample\*, [*Arnold: An Autonomous Agent to Play FPS Games*](http://www.cs.cmu.edu/~dchaplot/papers/arnold_aaai17.pdf)
```
@inproceedings{chaplot2017arnold,
  title={Arnold: An Autonomous Agent to Play FPS Games.},
  author={Chaplot, Devendra Singh and Lample, Guillaume},
  booktitle={Proceedings of AAAI},
  year={2017},
  Note={Best Demo award}
}
```

## Acknowledgements
We acknowledge the developers of [*ViZDoom*](http://vizdoom.cs.put.edu.pl/) for constant help and support during the development of this project. Some of the maps and wad files have been borrowed from the ViZDoom [*git repository*](https://github.com/mwydmuch/ViZDoom). We also thank the members of the [*ZDoom*](https://forum.zdoom.org/) community for their help with the Action Code Scripts (ACS).
