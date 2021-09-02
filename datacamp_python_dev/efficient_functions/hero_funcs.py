def convert_units(heroes, heights, weights):
    new_heights = [ht * 0.39370 for ht in heights]
    new_weights = [wt * 2.20462 for wt in weights]

    hero_data = {}

    for idx, hero in enumerate(heroes):
        hero_data[hero] = (new_heights[idx], new_weights[idx])
    return hero_data


def convert_units_broadcast(heroes, heights, weights):

    # Array broadcasting instead of list comprehension
    new_hts = heights * 0.39370
    new_wts = weights * 2.20462

    hero_data = {}

    for i,hero in enumerate(heroes):
        hero_data[hero] = (new_hts[i], new_wts[i])

    return hero_data