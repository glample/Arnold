def parse_labels_mapping(s):
    """
    Parse the mapping between a label type and it's feature map.
    For instance:
    '0;1;2;3' -> [0, 1, 2, 3]
    '0+2;3'   -> [0, None, 0, 1]
    '3;0+2;1' -> [1, 2, 1, 0]
    """
    if len(s) > 0:
        split = [[int(y) for y in x.split('+')] for x in s.split(';')]
        elements = sum(split, [])
        assert all(x in range(4) for x in elements)
        assert len(elements) == len(set(elements))
        labels_mapping = []
        for i in range(4):
            found = False
            for j, l in enumerate(split):
                if i in l:
                    assert not found
                    found = True
                    labels_mapping.append(j)
            if not found:
                labels_mapping.append(None)
        assert len(labels_mapping) == 4
    else:
        labels_mapping = None
    return labels_mapping


ENEMY_NAME_SET = set([
    'MarineBFG', 'MarineBerserk', 'MarineChaingun', 'MarineChainsaw',
    'MarineFist', 'MarinePistol', 'MarinePlasma', 'MarineRailgun',
    'MarineRocket', 'MarineSSG', 'MarineShotgun',
    'Demon'
])
HEALTH_ITEM_NAME_SET = set([
    'ArmorBonus', 'BlueArmor', 'GreenArmor', 'HealthBonus',
    'Medikit', 'Stimpack'
])
WEAPON_NAME_SET = set([
    'Pistol', 'Chaingun', 'RocketLauncher', 'Shotgun', 'SuperShotgun',
    'PlasmaRifle', 'BFG9000', 'Chainsaw'
])
AMMO_NAME_SET = set([
    'Cell', 'CellPack', 'Clip', 'ClipBox', 'RocketAmmo', 'RocketBox',
    'Shell', 'ShellBox'
])


def get_label_type_id(label):
    """
    Map an object name to a feature map.
    0 = enemy
    1 = health item
    2 = weapon
    3 = ammo
    None = anything else
    """
    name = label.object_name
    value = label.value
    if value != 255 and name == 'DoomPlayer' or name in ENEMY_NAME_SET:
        return 0
    elif name in HEALTH_ITEM_NAME_SET:
        return 1
    elif name in WEAPON_NAME_SET:
        return 2
    elif name in AMMO_NAME_SET:
        return 3
