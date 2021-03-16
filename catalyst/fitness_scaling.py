import numpy as np


def scale_scores(population, scaling_function, sa_screening):
    if sa_screening:
        for individual in population.molecules:
            if individual.energy == None:
                individual.score = 0
                individual.warnings += ['No Energy']
            else:
                individual.score = scaling_function(individual.energy) * individual.sa_score
    else:
        for individual in population.molecules:
            if individual.energy == None:
                individual.score = 0
                individual.warnings += ['No Energy']
            else:
                individual.score = scaling_function(individual.energy)

def linear_scaling(val, from_min=-1.5, from_max=1):
    to_min = 0
    to_max = 10
    scaled_value = (val - from_max) * (to_max - to_min) / -(from_max - from_min) + to_min
    if scaled_value > to_max:
        return to_max
    elif scaled_value < to_min:
        return to_min
    else:
        return scaled_value

def open_linear_scaling(val, from_max=60):
    to_min = 0
    to_max = 10
    scaled_value = (val - from_max) * (to_max - to_min) / -80 + to_min
    if scaled_value > to_max:
        return scaled_value
    elif scaled_value < to_min:
        return to_min
    else:
        return scaled_value

def exponential_scaling(val, b=0.1):
    return np.exp(-b*val)

def sigmoid_scaling(val, a=0.2, b=-15):
    scaled_value = 10/(1 + np.exp(a*(val-b)))
    return scaled_value