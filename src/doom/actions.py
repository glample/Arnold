import numpy as np
from vizdoom import Button
from logging import getLogger


# logger
logger = getLogger()


class ActionBuilder(object):

    def __init__(self, params):
        self.params = params
        self.available_buttons = get_available_buttons(self.params)
        logger.info('%i available buttons: %s' % (len(self.available_buttons),
                                                  str(self.available_buttons)))

        if params.use_continuous:
            assert params.speed == 'manual'
            self.available_actions = create_action_set(
                params.action_combinations,
                params.use_continuous
            )
            if params.crouch == 'manual':
                self.available_actions.append('CROUCH')
            # number of continuous / discrete actions
            params.n_continuous = sum(['_DELTA' in x
                                       for x in self.available_actions])
            params.n_discrete = (len(self.available_actions) -
                                 params.n_continuous)
        else:
            assert params.speed != 'manual' and params.crouch != 'manual'
            self.available_actions = create_action_set(
                params.action_combinations,
                params.use_continuous
            )
            # pre-compute ViZDoom actions
            self.doom_actions = []
            for sub_actions in self.available_actions:
                doom_action = [button in sub_actions
                               for button in self.available_buttons[:-2]]
                doom_action.append(params.speed == 'on')
                doom_action.append(params.crouch == 'on')
                self.doom_actions.append(doom_action)
        self.n_actions = len(self.available_actions)
        # In the continuous case, n_actions represents the number of values
        # to output to generate the complete set of actions.
        # For DELTA actions we generate 1 value (a mean), as well as for
        # discrete ones where we generate a probability (Bernoulli).
        params.n_actions = self.n_actions
        logger.info('%i available actions:\n%s' % (
            self.n_actions,
            '\n'.join(str(a) for a in self.available_actions)
        ))

    def get_action(self, action):
        """
        Convert selected action to the ViZDoom action format.
        """
        if self.params.use_continuous:
            assert type(action) is list
            assert len(action) == len(self.available_actions)
            _doom_action = {}
            for x, y in zip(action, self.available_actions):
                # TODO check delta ranges
                # TODO Bernoulli for discrete actions
                if y.startswith('MOVE'):  # in [-50, 50]
                    assert y.endswith('DELTA')
                    _doom_action[y] = int(np.clip(x, -1, 1) * 50)
                elif y.startswith('TURN'):  # in [-180, 180]
                    assert y.endswith('DELTA')
                    _doom_action[y] = int(np.clip(x, -1, 1) * 15)
                elif y.startswith('LOOK'):  # in [-X, X]
                    assert y.endswith('DELTA')
                    _doom_action[y] = int(np.clip(x, -1, 1) * 5)
                elif y == 'ATTACK':
                    assert x in [0, 1]
                    _doom_action[y] = bool(x)
                elif y == 'CROUCH':
                    assert x in [0, 1] and self.params.crouch == 'manual'
                    _doom_action[y] = bool(x)
                else:
                    raise Exception('Unexpected action: "%s"' % y)
            if self.params.crouch != 'manual':
                assert 'CROUCH' not in _doom_action
                _doom_action['CROUCH'] = self.params.crouch == 'on'
            doom_action = [_doom_action[k] if k in _doom_action else 0
                           for k in self.available_buttons]
            return doom_action
        else:
            assert type(action) is int
            return self.doom_actions[action]


action_categories_discrete = {
    'turn_lr': ['TURN_LEFT', 'TURN_RIGHT'],
    'look_ud': ['LOOK_UP', 'LOOK_DOWN'],
    'move_fb': ['MOVE_FORWARD', 'MOVE_BACKWARD'],
    'move_lr': ['MOVE_LEFT', 'MOVE_RIGHT'],
    'attack': ['ATTACK'],
}

action_categories_continuous = {
    'turn_lr': 'TURN_LEFT_RIGHT_DELTA',
    'look_ud': 'LOOK_UP_DOWN_DELTA',
    'move_fb': 'MOVE_FORWARD_BACKWARD_DELTA',
    'move_lr': 'MOVE_LEFT_RIGHT_DELTA',
    'attack': 'ATTACK',
}


def create_action_set(action_combinations, use_continuous):
    """
    Create the set of possible actions given the allowed action combinations.
    An action is a combination of one or several buttons.
    The '+' merges buttons that should not be triggered together.
    The ';' separates groups of buttons that can be triggered simultaneously.
    For example:
    Input:
        'turn_lr+move_fb;move_lr'
    Output:
        [['MOVE_LEFT'],
         ['MOVE_RIGHT'],
         ['TURN_LEFT'],
         ['TURN_LEFT', 'MOVE_LEFT'],
         ['TURN_LEFT', 'MOVE_RIGHT'],
         ['TURN_RIGHT'],
         ['TURN_RIGHT', 'MOVE_LEFT'],
         ['TURN_RIGHT', 'MOVE_RIGHT'],
         ['MOVE_FORWARD'],
         ['MOVE_FORWARD', 'MOVE_LEFT'],
         ['MOVE_FORWARD', 'MOVE_RIGHT'],
         ['MOVE_BACKWARD'],
         ['MOVE_BACKWARD', 'MOVE_LEFT'],
         ['MOVE_BACKWARD', 'MOVE_RIGHT']]
    In continuous mode, all actions can be selected simultaneously, so there
    should be no "+" in the action combinations.
    """
    if use_continuous:
        assert '+' not in action_combinations
        action_set = [action_categories_continuous[x]
                      for x in action_combinations.split(';')]
        # check that the discrete actions are the last ones
        delta = ['_DELTA' in x for x in action_set]
        assert all(x or (not any(delta[i + 1:])) for i, x in enumerate(delta))
    else:
        action_subsets = [
            sum([action_categories_discrete[y] for y in x.split('+')], [None])
            for x in action_combinations.split(';')
        ]
        action_set = [[]]
        for subset2 in action_subsets:
            action_set = sum([[subset1 + [x] for x in subset2]
                              for subset1 in action_set], [])
        action_set = [[y for y in x if y is not None] for x in action_set]
        action_set = [z for z in action_set if len(z) > 0]
    return action_set


def get_available_buttons(params):
    """
    Create a list of all buttons available to the agent.
    """
    available_buttons = []

    # move + turn (+ optional look up / look down)
    if params.use_continuous:
        available_buttons.append('MOVE_FORWARD_BACKWARD_DELTA')
        available_buttons.append('TURN_LEFT_RIGHT_DELTA')
        available_buttons.append('MOVE_LEFT_RIGHT_DELTA')
        if params.freelook:
            available_buttons.append('LOOK_UP_DOWN_DELTA')
    else:
        available_buttons.extend(['MOVE_FORWARD', 'MOVE_BACKWARD'])
        available_buttons.extend(['TURN_LEFT', 'TURN_RIGHT'])
        available_buttons.extend(['MOVE_LEFT', 'MOVE_RIGHT'])
        if params.freelook:
            available_buttons.extend(['LOOK_UP', 'LOOK_DOWN'])

    # attack is always discrete
    available_buttons.append('ATTACK')

    # speed is only for the non-continuous mode
    if not params.use_continuous:
        available_buttons.append('SPEED')

    # crouch is always discrete
    available_buttons.append('CROUCH')

    return available_buttons


def add_buttons(game, buttons):
    """
    Add all available buttons to the game.
    """
    # add buttons to select different weapons
    for i in range(10):
        buttons.append("SELECT_WEAPON%i" % i)

    for s in buttons:

        s = s.lower()

        if (s == "attack"):
            game.add_available_button(Button.ATTACK)
        elif (s == "use"):
            game.add_available_button(Button.USE)
        elif (s == "jump"):
            game.add_available_button(Button.JUMP)
        elif (s == "crouch"):
            game.add_available_button(Button.CROUCH)
        elif (s == "turn180"):
            game.add_available_button(Button.TURN180)
        elif (s == "alattack"):
            game.add_available_button(Button.ALTATTACK)
        elif (s == "reload"):
            game.add_available_button(Button.RELOAD)
        elif (s == "zoom"):
            game.add_available_button(Button.ZOOM)
        elif (s == "speed"):
            game.add_available_button(Button.SPEED)
        elif (s == "strafe"):
            game.add_available_button(Button.STRAFE)
        elif (s == "move_right"):
            game.add_available_button(Button.MOVE_RIGHT)
        elif (s == "move_left"):
            game.add_available_button(Button.MOVE_LEFT)
        elif (s == "move_backward"):
            game.add_available_button(Button.MOVE_BACKWARD)
        elif (s == "move_forward"):
            game.add_available_button(Button.MOVE_FORWARD)
        elif (s == "turn_right"):
            game.add_available_button(Button.TURN_RIGHT)
        elif (s == "turn_left"):
            game.add_available_button(Button.TURN_LEFT)
        elif (s == "look_up"):
            game.add_available_button(Button.LOOK_UP)
        elif (s == "look_down"):
            game.add_available_button(Button.LOOK_DOWN)
        elif (s == "move_up"):
            game.add_available_button(Button.MOVE_UP)
        elif (s == "move_down"):
            game.add_available_button(Button.MOVE_DOWN)
        elif (s == "land"):
            game.add_available_button(Button.LAND)

        elif (s == "select_weapon1"):
            game.add_available_button(Button.SELECT_WEAPON1)
        elif (s == "select_weapon2"):
            game.add_available_button(Button.SELECT_WEAPON2)
        elif (s == "select_weapon3"):
            game.add_available_button(Button.SELECT_WEAPON3)
        elif (s == "select_weapon4"):
            game.add_available_button(Button.SELECT_WEAPON4)
        elif (s == "select_weapon5"):
            game.add_available_button(Button.SELECT_WEAPON5)
        elif (s == "select_weapon6"):
            game.add_available_button(Button.SELECT_WEAPON6)
        elif (s == "select_weapon7"):
            game.add_available_button(Button.SELECT_WEAPON7)
        elif (s == "select_weapon8"):
            game.add_available_button(Button.SELECT_WEAPON8)
        elif (s == "select_weapon9"):
            game.add_available_button(Button.SELECT_WEAPON9)
        elif (s == "select_weapon0"):
            game.add_available_button(Button.SELECT_WEAPON0)

        elif (s == "select_next_weapon"):
            game.add_available_button(Button.SELECT_NEXT_WEAPON)
        elif (s == "select_prev_weapon"):
            game.add_available_button(Button.SELECT_PREV_WEAPON)
        elif (s == "drop_selected_weapon"):
            game.add_available_button(Button.DROP_SELECTED_WEAPON)
        elif (s == "activate_selected_weapon"):
            game.add_available_button(Button.ACTIVATE_SELECTED_ITEM)
        elif (s == "select_next_item"):
            game.add_available_button(Button.SELECT_NEXT_ITEM)
        elif (s == "select_prev_item"):
            game.add_available_button(Button.SELECT_PREV_ITEM)
        elif (s == "drop_selected_item"):
            game.add_available_button(Button.DROP_SELECTED_ITEM)

        elif (s == "look_up_down_delta"):
            game.add_available_button(Button.LOOK_UP_DOWN_DELTA)
        elif (s == "turn_left_right_delta"):
            game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
        elif (s == "move_forward_backward_delta"):
            game.add_available_button(Button.MOVE_FORWARD_BACKWARD_DELTA)
        elif (s == "move_left_right_delta"):
            game.add_available_button(Button.MOVE_LEFT_RIGHT_DELTA)
        elif (s == "move_up_down_delta"):
            game.add_available_button(Button.MOVE_UP_DOWN_DELTA)

        else:
            raise Exception("Unknown button!")

    return {k: i for i, k in enumerate(buttons)}
