import os
import time
import math
from logging import getLogger
from collections import namedtuple

# ViZDoom library
from vizdoom import DoomGame, GameVariable
from vizdoom import ScreenResolution, ScreenFormat, Mode

# Arnold
from .utils import process_buffers
from .reward import RewardBuilder
from .actions import add_buttons
from .labels import parse_labels_mapping
from .game_features import parse_game_features


RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources')

WEAPON_NAMES = [None, "Fist", "Pistol", "SuperShotgun", "Chaingun",
                "RocketLauncher", "PlasmaRifle", "BFG9000"]

WEAPONS_PREFERENCES = [
    ('bfg9000', 'cells', 7), ('shotgun', 'shells', 3),
    ('chaingun', 'bullets', 4), ('plasmarifle', 'cells', 6),
    ('pistol', 'bullets', 2), ('rocketlauncher', 'rockets', 5)
]

RESPAWN_SECONDS = 10

# game variables we want to use
game_variables = [
    # ('KILLCOUNT', GameVariable.KILLCOUNT),
    # ('ITEMCOUNT', GameVariable.ITEMCOUNT),
    # ('SECRETCOUNT', GameVariable.SECRETCOUNT),
    ('frag_count', GameVariable.FRAGCOUNT),
    # ('DEATHCOUNT', GameVariable.DEATHCOUNT),
    ('health', GameVariable.HEALTH),
    ('armor', GameVariable.ARMOR),
    # ('DEAD', GameVariable.DEAD),
    # ('ON_GROUND', GameVariable.ON_GROUND),
    # ('ATTACK_READY', GameVariable.ATTACK_READY),
    # ('ALTATTACK_READY', GameVariable.ALTATTACK_READY),
    ('sel_weapon', GameVariable.SELECTED_WEAPON),
    ('sel_ammo', GameVariable.SELECTED_WEAPON_AMMO),
    # ('AMMO0', GameVariable.AMMO0),  # UNK
    # ('AMMO1', GameVariable.AMMO1),  # fist weapon, should always be 0
    ('bullets', GameVariable.AMMO2),  # bullets
    ('shells', GameVariable.AMMO3),  # shells
    # ('AMMO4', GameVariable.AMMO4),  # == AMMO2
    ('rockets', GameVariable.AMMO5),  # rockets
    ('cells', GameVariable.AMMO6),  # cells
    # ('AMMO7', GameVariable.AMMO7),  # == AMMO6
    # ('AMMO8', GameVariable.AMMO8),  # UNK
    # ('AMMO9', GameVariable.AMMO9),  # UNK
    # ('WEAPON0', GameVariable.WEAPON0),  # UNK
    ('fist', GameVariable.WEAPON1),  # Fist, should be 1, unless removed
    ('pistol', GameVariable.WEAPON2),  # Pistol
    ('shotgun', GameVariable.WEAPON3),  # Shotgun
    ('chaingun', GameVariable.WEAPON4),  # Chaingun
    ('rocketlauncher', GameVariable.WEAPON5),  # Rocket Launcher
    ('plasmarifle', GameVariable.WEAPON6),  # Plasma Rifle
    ('bfg9000', GameVariable.WEAPON7),  # BFG9000
    # ('WEAPON8', GameVariable.WEAPON8),  # UNK
    # ('WEAPON9', GameVariable.WEAPON9),  # UNK
    ('position_x', GameVariable.POSITION_X),
    ('position_y', GameVariable.POSITION_Y),
    ('position_z', GameVariable.POSITION_Z),
    # ('velocity_x', GameVariable.VELOCITY_X),
    # ('velocity_y', GameVariable.VELOCITY_Y),
    # ('velocity_z', GameVariable.VELOCITY_Z),
]

# advance a few steps to avoid bugs due to initial weapon changes
SKIP_INITIAL_ACTIONS = 3


# game state
GameState = namedtuple('State', ['screen', 'variables', 'features'])


# logger
logger = getLogger()


class Game(object):

    def __init__(
        self,
        scenario,
        action_builder,
        reward_values=None,
        score_variable='FRAGCOUNT',
        freedoom=True,
        screen_resolution='RES_400X225',
        screen_format='CRCGCB',
        use_screen_buffer=True,
        use_depth_buffer=False,
        labels_mapping='',
        game_features='',
        mode='PLAYER',
        player_rank=0, players_per_game=1,
        render_hud=False, render_minimal_hud=False,
        render_crosshair=True, render_weapon=True,
        render_decals=False,
        render_particles=False,
        render_effects_sprites=False,
        respawn_protect=True, spawn_farthest=True,
        freelook=False, name='Arnold', color=0,
        visible=False,
        n_bots=0, use_scripted_marines=None,
        doom_skill=2
    ):
        """
        Create a new game.
        score_variable: indicates in which game variable the user score is
            stored. by default it's in FRAGCOUNT, but the score in ACS against
            built-in AI bots can be stored in USER1, USER2, etc.
        render_decals: marks on the walls
        render_particles: particles like for impacts / traces
        render_effects_sprites: gun puffs / blood splats
        color: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray,
               5 - light brown, 6 - light red, 7 - light blue
        """
        # game resources
        game_filename = '%s.wad' % ('freedoom2' if freedoom else 'Doom2')
        self.scenario_path = os.path.join(RESOURCES_DIR, 'scenarios', '%s.wad' % scenario)
        self.game_path = os.path.join(RESOURCES_DIR, game_filename)

        # check parameters
        assert os.path.isfile(self.scenario_path)
        assert os.path.isfile(self.game_path)
        assert hasattr(GameVariable, score_variable)
        assert hasattr(ScreenResolution, screen_resolution)
        assert hasattr(ScreenFormat, screen_format)
        assert use_screen_buffer or use_depth_buffer
        assert hasattr(Mode, mode)
        assert not (render_minimal_hud and not render_hud)
        assert len(name.strip()) > 0 and color in range(8)
        assert n_bots >= 0
        assert (type(use_scripted_marines) is bool or
                use_scripted_marines is None and n_bots == 0)
        assert 0 <= doom_skill <= 4
        assert 0 < players_per_game
        assert 0 <= player_rank

        # action builder
        self.action_builder = action_builder

        # add the score variable to the game variables list
        self.score_variable = score_variable
        game_variables.append(('score', getattr(GameVariable, score_variable)))

        self.player_rank = player_rank
        self.players_per_game = players_per_game

        # screen buffer / depth buffer / labels buffer / mode
        self.screen_resolution = screen_resolution
        self.screen_format = screen_format
        self.use_screen_buffer = use_screen_buffer
        self.use_depth_buffer = use_depth_buffer
        self.labels_mapping = parse_labels_mapping(labels_mapping)
        self.game_features = parse_game_features(game_features)
        self.use_labels_buffer = self.labels_mapping is not None
        self.use_game_features = any(self.game_features)
        self.mode = mode

        # rendering options
        self.render_hud = render_hud
        self.render_minimal_hud = render_minimal_hud
        self.render_crosshair = render_crosshair
        self.render_weapon = render_weapon
        self.render_decals = render_decals
        self.render_particles = render_particles
        self.render_effects_sprites = render_effects_sprites

        # respawn invincibility / distance
        self.respawn_protect = respawn_protect
        self.spawn_farthest = spawn_farthest

        # freelook / agent name / agent color
        self.freelook = freelook
        self.name = name.strip()
        self.color = color

        # window visibility
        self.visible = visible

        # actor reward
        self.reward_builder = RewardBuilder(self, reward_values)

        # game statistics
        self.stat_keys = ['kills', 'deaths', 'suicides', 'frags', 'k/d',
                          'medikits', 'armors',
                          'pistol', 'shotgun', 'chaingun',
                          'rocketlauncher', 'plasmarifle', 'bfg9000',
                          'bullets', 'shells', 'rockets', 'cells']
        self.statistics = {}

        # number of bots in the game
        self.n_bots = n_bots
        self.use_scripted_marines = use_scripted_marines

        # doom skill
        self.doom_skill = doom_skill

        # manual control
        self.count_non_forward_actions = 0
        self.count_non_turn_actions = 0

    def update_game_variables(self):
        """
        Check and update game variables.
        """
        # read game variables
        new_v = {k: self.game.get_game_variable(v) for k, v in game_variables}
        assert all(v.is_integer() or k[-2:] in ['_x', '_y', '_z'] for k, v in new_v.items())
        new_v = {k: (int(v) if v.is_integer() else float(v)) for k, v in new_v.items()}
        health = new_v['health']
        armor = new_v['armor']
        sel_weapon = new_v['sel_weapon']
        sel_ammo = new_v['sel_ammo']
        bullets = new_v['bullets']
        shells = new_v['shells']
        rockets = new_v['rockets']
        cells = new_v['cells']
        fist = new_v['fist']
        pistol = new_v['pistol']
        shotgun = new_v['shotgun']
        chaingun = new_v['chaingun']
        rocketlauncher = new_v['rocketlauncher']
        plasmarifle = new_v['plasmarifle']
        bfg9000 = new_v['bfg9000']

        # check game variables
        if sel_weapon == -1:
            logger.warning("SELECTED WEAPON is -1!")
            new_v['sel_weapon'] = 1
            sel_weapon = 1
        if sel_ammo == -1:
            logger.warning("SELECTED AMMO is -1!")
            new_v['sel_ammo'] = 0
            sel_ammo = 0
        assert sel_weapon in range(1, 8), sel_weapon
        assert sel_ammo >= 0, sel_ammo
        assert all(x in [0, 1] for x in [fist, pistol, shotgun, chaingun,
                                         rocketlauncher, plasmarifle, bfg9000])
        assert 0 <= health <= 200 or health < 0 and self.game.is_player_dead()
        assert 0 <= armor <= 200, (health, armor)
        assert 0 <= bullets <= 200 and 0 <= shells <= 50
        assert 0 <= rockets <= 50 and 0 <= cells <= 300

        # fist
        if sel_weapon == 1:
            assert sel_ammo == 0
        # pistol
        elif sel_weapon == 2:
            assert pistol and sel_ammo == bullets
        # shotgun
        elif sel_weapon == 3:
            assert shotgun and sel_ammo == shells
        # chaingun
        elif sel_weapon == 4:
            assert chaingun and sel_ammo == bullets
        # rocket launcher
        elif sel_weapon == 5:
            assert rocketlauncher and sel_ammo == rockets
        # plasma rifle
        elif sel_weapon == 6:
            assert plasmarifle and sel_ammo == cells
        # BFG9000
        elif sel_weapon == 7:
            assert bfg9000 and sel_ammo == cells

        # update actor properties
        self.prev_properties = self.properties
        self.properties = new_v

    def update_statistics_and_reward(self, action):
        """
        Update statistics of the current game based on the previous
        and the current properties, and create a reward.
        """
        stats = self.statistics[self.map_id]

        # reset reward
        self.reward_builder.reset()

        # we need to know the current and previous properties
        assert self.prev_properties is not None and self.properties is not None

        # distance
        moving_forward = action[self.mapping['MOVE_FORWARD']]
        turn_left = action[self.mapping['TURN_LEFT']]
        turn_right = action[self.mapping['TURN_RIGHT']]
        if moving_forward and not (turn_left or turn_right):
            diff_x = self.properties['position_x'] - self.prev_properties['position_x']
            diff_y = self.properties['position_y'] - self.prev_properties['position_y']
            distance = math.sqrt(diff_x ** 2 + diff_y ** 2)
            self.reward_builder.distance(distance)

        # kill
        d = self.properties['score'] - self.prev_properties['score']
        if d > 0:
            self.reward_builder.kill(d)
            stats['kills'] += d
            for _ in range(int(d)):
                self.log('Kill')

        # death
        if self.game.is_player_dead():
            self.reward_builder.death()
            stats['deaths'] += 1
            self.log('Dead')

        # suicide
        if self.properties['frag_count'] < self.prev_properties['frag_count']:
            self.reward_builder.suicide()
            stats['suicides'] += 1
            self.log('Suicide')

        # found / lost health
        d = self.properties['health'] - self.prev_properties['health']
        if d != 0:
            if d > 0:
                self.reward_builder.medikit(d)
                stats['medikits'] += 1
            else:
                self.reward_builder.injured(d)
            self.log('%s health (%i -> %i)' % (
                'Found' if d > 0 else 'Lost',
                self.prev_properties['health'],
                self.properties['health'],
            ))

        # found / lost armor
        d = self.properties['armor'] - self.prev_properties['armor']
        if d != 0:
            if d > 0:
                self.reward_builder.armor()
                stats['armors'] += 1
            self.log('%s armor (%i -> %i)' % (
                'Found' if d > 0 else 'Lost',
                self.prev_properties['armor'],
                self.properties['armor'],
            ))

        # change weapon
        if self.properties['sel_weapon'] != self.prev_properties['sel_weapon']:
            self.log('Switched weapon: %s -> %s' % (
                WEAPON_NAMES[self.prev_properties['sel_weapon']],
                WEAPON_NAMES[self.properties['sel_weapon']],
            ))

        # found weapon
        for i, weapon in enumerate(['pistol', 'shotgun', 'chaingun',
                                    'rocketlauncher', 'plasmarifle',
                                    'bfg9000']):
            if self.prev_properties[weapon] == self.properties[weapon]:
                continue
            # assert(self.prev_properties[weapon] == 0 and  # TODO check
            #        self.properties[weapon] == 1), (weapon, self.prev_properties[weapon], self.properties[weapon])
            self.reward_builder.weapon()
            stats[weapon] += 1
            self.log('Found weapon: %s' % WEAPON_NAMES[i + 1])

        # found / lost ammo
        for ammo in ['bullets', 'shells', 'rockets', 'cells']:
            d = self.properties[ammo] - self.prev_properties[ammo]
            if d != 0:
                if d > 0:
                    self.reward_builder.ammo()
                    stats[ammo] += 1
                else:
                    self.reward_builder.use_ammo()
                self.log('%s ammo: %s (%i -> %i)' % (
                    'Found' if d > 0 else 'Lost',
                    ammo,
                    self.prev_properties[ammo],
                    self.properties[ammo]
                ))

    def log(self, message):
        """
        Log the game event.
        During training, we don't want to display events.
        """
        if self.log_events:
            logger.info(message)

    def start(self, map_id, episode_time=None, manual_control=False, log_events=False):
        """
        Start the game.
        If `episode_time` is given, the game will end after the specified time.
        """
        assert type(manual_control) is bool
        self.manual_control = manual_control

        # Save statistics for this map
        self.statistics[map_id] = {k: 0 for k in self.stat_keys}

        # Episode time
        self.episode_time = episode_time

        # initialize the game
        self.game = DoomGame()
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_doom_game_path(self.game_path)

        # map
        assert map_id > 0
        self.map_id = map_id
        self.game.set_doom_map("map%02i" % map_id)

        # time limit
        if episode_time is not None:
            self.game.set_episode_timeout(int(35 * episode_time))

        # log events that happen during the game (useful for testing)
        self.log_events = log_events

        # game parameters
        args = []

        # host / server
        if self.players_per_game > 1:
            port = 5092 + self.player_rank // self.players_per_game
            if self.player_rank % self.players_per_game == 0:
                args.append('-host %i -port %i' % (self.players_per_game, port))
            else:
                args.append('-join 127.0.0.1:%i' % port)
        else:
            args.append('-host 1')

        # screen buffer / depth buffer / labels buffer / mode
        screen_resolution = getattr(ScreenResolution, self.screen_resolution)
        self.game.set_screen_resolution(screen_resolution)
        self.game.set_screen_format(getattr(ScreenFormat, self.screen_format))
        self.game.set_depth_buffer_enabled(self.use_depth_buffer)
        self.game.set_labels_buffer_enabled(self.use_labels_buffer or
                                            self.use_game_features)
        self.game.set_mode(getattr(Mode, self.mode))

        # rendering options
        self.game.set_render_hud(self.render_hud)
        self.game.set_render_minimal_hud(self.render_minimal_hud)
        self.game.set_render_crosshair(self.render_crosshair)
        self.game.set_render_weapon(self.render_weapon)
        self.game.set_render_decals(self.render_decals)
        self.game.set_render_particles(self.render_particles)
        self.game.set_render_effects_sprites(self.render_effects_sprites)

        # deathmatch mode
        # players will respawn automatically after they die
        # autoaim is disabled for all players
        args.append('-deathmatch')
        args.append('+sv_forcerespawn 1')
        args.append('+sv_noautoaim 1')

        # respawn invincibility / distance
        # players will be invulnerable for two second after spawning
        # players will be spawned as far as possible from any other players
        args.append('+sv_respawnprotect %i' % self.respawn_protect)
        args.append('+sv_spawnfarthest %i' % self.spawn_farthest)

        # freelook / agent name / agent color
        args.append('+freelook %i' % (1 if self.freelook else 0))
        args.append('+name %s' % self.name)
        args.append('+colorset %i' % self.color)

        # enable the cheat system (so that we can still
        # send commands to the game in self-play mode)
        args.append('+sv_cheats 1')

        # load parameters
        self.args = args
        for arg in args:
            self.game.add_game_args(arg)

        # window visibility
        self.game.set_window_visible(self.visible)

        # available buttons
        self.mapping = add_buttons(self.game, self.action_builder.available_buttons)

        # doom skill (https://zdoom.org/wiki/GameSkill)
        self.game.set_doom_skill(self.doom_skill + 1)

        # start the game
        self.game.init()

        # initialize the game after player spawns
        self.initialize_game()

    def reset(self):
        """
        Reset the game if necessary. This can be because:
            - we reach the end of an episode (we restart the game)
            - because the agent is dead (we make it respawn)
        """
        self.count_non_forward_actions = 0
        # if the player is dead
        if self.is_player_dead():
            # respawn it (deathmatch mode)
            if self.episode_time is None:
                self.respawn_player()
            # or reset the episode (episode ends when the agent dies)
            else:
                self.new_episode()

        # start a new episode if it is finished
        if self.is_episode_finished():
            self.new_episode()

        # deal with a ViZDoom issue
        while self.is_player_dead():
            logger.warning('Player %i is still dead after respawn.' %
                           self.params.player_rank)
            self.respawn_player()

    def update_bots(self):
        """
        Add built-in AI bots.
        There are two types of AI: built-in AI and ScriptedMarines.
        """
        # only the host takes care of the bots
        if self.player_rank % self.players_per_game != 0:
            return
        if self.use_scripted_marines:
            command = "pukename set_value always 2 %i" % self.n_bots
            self.game.send_game_command(command)
        else:
            self.game.send_game_command("removebots")
            for _ in range(self.n_bots):
                self.game.send_game_command("addbot")

    def is_player_dead(self):
        """
        Detect whether the player is dead.
        """
        return self.game.is_player_dead()

    def is_episode_finished(self):
        """
        Return whether the episode is finished.
        This should only be the case after the episode timeout.
        """
        return self.game.is_episode_finished()

    def is_final(self):
        """
        Return whether the game is in a final state.
        """
        return self.is_player_dead() or self.is_episode_finished()

    def new_episode(self):
        """
        Start a new episode.
        """
        assert self.is_episode_finished() or self.is_player_dead()
        self.game.new_episode()
        self.log('New episode')
        self.initialize_game()

    def respawn_player(self):
        """
        Respawn the player on death.
        """
        assert self.is_player_dead()
        self.game.respawn_player()
        self.log('Respawn player')
        self.initialize_game()

    def initialize_game(self):
        """
        Initialize the game after the player spawns / respawns.
        Be sure that properties from the previous
        life are not considered in this one.
        """
        # generate buffers
        game_state = self.game.get_state()
        self._screen_buffer = game_state.screen_buffer
        self._depth_buffer = game_state.depth_buffer
        self._labels_buffer = game_state.labels_buffer
        self._labels = game_state.labels

        # actor properties
        self.prev_properties = None
        self.properties = None

        # advance a few steps to avoid bugs due
        # to initial weapon changes in ACS
        self.game.advance_action(SKIP_INITIAL_ACTIONS)
        self.update_game_variables()

        # if there are bots in the game, and if this is a new game
        self.update_bots()

    def randomize_textures(self, randomize):
        """
        Randomize the textures of the map.
        """
        assert type(randomize) is bool
        randomize = 1 if randomize else 0
        self.game.send_game_command("pukename set_value always 4 %i" % randomize)

    def init_bots_health(self, health):
        """
        Initial bots health.
        """
        assert self.use_scripted_marines or health == 100
        assert 0 < health <= 100
        self.game.send_game_command("pukename set_value always 5 %i" % health)

    def make_action(self, action, frame_skip=1, sleep=None):
        """
        Make an action.
        If `sleep` is given, the network will wait
        `sleep` seconds between each action.
        """
        assert frame_skip >= 1

        # convert selected action to the ViZDoom action format
        action = self.action_builder.get_action(action)

        # select agent favorite weapon
        for weapon_name, weapon_ammo, weapon_id in WEAPONS_PREFERENCES:
            min_ammo = 40 if weapon_name == 'bfg9000' else 1
            if self.properties[weapon_name] > 0 and self.properties[weapon_ammo] >= min_ammo:
                if self.properties['sel_weapon'] != weapon_id:
                    # action = ([False] * self.mapping['SELECT_WEAPON%i' % weapon_id]) + [True]
                    switch_action = ([False] * self.mapping['SELECT_WEAPON%i' % weapon_id]) + [True]
                    action = action + switch_action[len(action):]
                    self.log("Manual weapon change: %s -> %s" % (WEAPON_NAMES[self.properties['sel_weapon']], weapon_name))
                break

        if action[self.mapping['MOVE_FORWARD']]:
            self.count_non_forward_actions = 0
        else:
            self.count_non_forward_actions += 1

        if action[self.mapping['TURN_LEFT']] or action[self.mapping['TURN_RIGHT']]:
            self.count_non_turn_actions = 0
        else:
            self.count_non_turn_actions += 1

        if self.manual_control and (self.count_non_forward_actions >= 30 or self.count_non_turn_actions >= 60):
            manual_action = [False] * len(action)
            manual_action[self.mapping['TURN_RIGHT']] = True
            manual_action[self.mapping['SPEED']] = True
            if self.count_non_forward_actions >= 30:
                manual_action[self.mapping['MOVE_FORWARD']] = True
            manual_repeat = 40
            self.count_non_forward_actions = 0
            self.count_non_turn_actions = 0
        else:
            manual_action = None

        # if we are visualizing the experiment, show all the frames one by one
        if self.visible:
            if manual_action is not None:
                logger.warning('Activated manual control')
                for _ in range(manual_repeat):
                    self.game.make_action(manual_action)
            else:
                for _ in range(frame_skip):
                    self.game.make_action(action)
                    # death or episode finished
                    if self.is_player_dead() or self.is_episode_finished():
                        break
                    # sleep for smooth visualization
                    if sleep is not None:
                        time.sleep(sleep)
        else:
            if manual_action is not None:
                logger.warning('Activated manual control')
                self.game.make_action(manual_action, manual_repeat)
            else:
                self.game.make_action(action, frame_skip)

        # generate buffers
        game_state = self.game.get_state()
        if game_state is not None:
            self._screen_buffer = game_state.screen_buffer
            self._depth_buffer = game_state.depth_buffer
            self._labels_buffer = game_state.labels_buffer
            self._labels = game_state.labels

        # update game variables / statistics rewards
        self.update_game_variables()
        self.update_statistics_and_reward(action)

    @property
    def reward(self):
        """
        Return the reward value.
        """
        return self.reward_builder.reward

    def close(self):
        """
        Close the current game.
        """
        self.game.close()

    def print_statistics(self, eval_time=None):
        """
        Print agent statistics.
        If `map_id` is given, statistics are given for the specified map only.
        Otherwise, statistics are given for all maps, with a summary.
        """
        if 'all' in self.statistics:
            del self.statistics['all']
        map_ids = sorted(self.statistics.keys())
        if len(map_ids) == 0:
            logger.info("No statistics to show!")
            return
        for v in self.statistics.values():
            assert set(self.stat_keys) == set(v.keys())

        # sum the results on all maps for global statistics
        self.statistics['all'] = {
            k: sum(v[k] for v in self.statistics.values())
            for k in self.stat_keys
        }

        # number of frags (kills - suicides)
        # 100% accurate if the number of frags is given by 'FRAGCOUNT'
        # almost 100% accurate if it is based on an internal ACS variable
        for v in self.statistics.values():
            v['frags'] = v['kills'] - v['suicides']

        # number of frags per minutes (with and without respawn time)
        if eval_time is not None:
            assert eval_time % 60 == 0
            for k, v in self.statistics.items():
                eval_minutes = eval_time / 60
                if k == 'all':
                    eval_minutes *= (len(self.statistics) - 1)
                respawn_time = (v['deaths'] * RESPAWN_SECONDS * 1.0 / 60)
                v['frags_pm'] = v['frags'] * 1.0 / eval_minutes
                v['frags_pm_r'] = v['frags'] * 1.0 / (eval_minutes + respawn_time)

        # Kills / Deaths
        # 100% accurate if the number of kills is given by an ACS variable
        # almost 100% accurate if it is based on 'FRAGCOUNT'
        for v in self.statistics.values():
            v['k/d'] = v['kills'] * 1.0 / max(1, v['deaths'])

        # statistics to log
        log_lines = [
            [''] + ['Map%02i' % i for i in map_ids] + ['All'],
            ('Kills', 'kills'),
            ('Deaths', 'deaths'),
            ('Suicides', 'suicides'),
            ('Frags', 'frags'),
            ('Frags/m', 'frags_pm'),
            ('Frags/m (r)', 'frags_pm_r'),
            ('K/D', 'k/d'),
            None,
            ('Medikits', 'medikits'),
            ('Armors', 'armors'),
            ('SuperShotgun', 'shotgun'),
            ('Chaingun', 'chaingun'),
            ('RocketLauncher', 'rocketlauncher'),
            ('PlasmaRifle', 'plasmarifle'),
            ('BFG9000', 'bfg9000'),
            ('Bullets', 'bullets'),
            ('Shells', 'shells'),
            ('Rockets', 'rockets'),
            ('Cells', 'cells'),
        ]

        # only show statistics on all maps if there is more than one map
        if len(map_ids) > 1:
            map_ids.append('all')

        logger.info('*************** Game statistics summary ***************')
        log_pattern = '{: >15}' + ('{: >8}' * len(map_ids))
        for line in log_lines:
            if line is None:
                logger.info('')
            else:
                if type(line) is tuple:
                    assert len(line) == 2
                    name, k = line
                    if k in ['frags_pm', 'frags_pm_r'] and eval_time is None:
                        continue
                    line = ['%s:' % name]
                    line += [self.statistics[map_id][k] for map_id in map_ids]
                else:
                    assert type(line) is list
                    line = line[:len(map_ids) + 1]
                line = ['%.3f' % x if type(x) is float else x for x in line]
                logger.info(log_pattern.format(*line))

    def observe_state(self, params, last_states):
        """
        Observe the current state of the game.
        """
        # read game state
        screen, game_features = process_buffers(self, params)
        variables = [self.properties[x[0]] for x in params.game_variables]
        last_states.append(GameState(screen, variables, game_features))

        # update most recent states
        if len(last_states) == 1:
            last_states.extend([last_states[0]] * (params.hist_size - 1))
        else:
            assert len(last_states) == params.hist_size + 1
            del last_states[0]

        # return the screen and the game features
        return screen, game_features
