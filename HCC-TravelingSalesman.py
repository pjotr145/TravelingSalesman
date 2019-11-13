#!/usr/bin/env python
"""Namaken van basic versie van het Traveling Salesman probleem"""

from math import sqrt
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
#import pdb; pdb.set_trace()

# Verschillende constanten
AANTSTEDEN = 51
SIZEPOPULATION = 500
MUTATIONRATE = 0.01
FITNESSTOTHEPOWER = 3
NUMBEROFGENERATIONS = 601
TRIPSHOULDBELOOP = True
# cities oriented in a "circular", "twocircles" or "random" shape.
CITIESORIENTATION = "twocircles"
RADIUS = 1
RADIUSTWO = [1, 1.1]
CENTRE = [0, 0]  # Not used yet
# chromosome chosen "roulette" from fitnesslist, "eliteist" or "superras"
SELECTIONCRITERIUM = "superras"
# method to fill when using superras: "onesplit", "twosplit", "random", "all"
SUPERMETHOD = "all"
SUPERALLCHANCES = [0.7, 0.9]

def create_cities():
    """Create AANTSTEDEN amount of x,y combinations.

    Orientation is either "circular" or "random"
    output is a numpy array
    """
    if CITIESORIENTATION == "circular":
        radials = np.arange(0, (2 * np.pi), (2 * np.pi) / AANTSTEDEN)
        x_coord = RADIUS * np.cos(radials)
        y_coord = RADIUS * np.sin(radials)
        cities = np.stack((x_coord, y_coord), axis=-1)
    elif CITIESORIENTATION == "twocircles":
        # if AANTSTEDEN not an even number, make sure both parts are integers
        aant_1 = int(AANTSTEDEN / 2)
        aant_2 = AANTSTEDEN - aant_1
        radials_1 = np.arange(0, (2 * np.pi), (2 * np.pi) / aant_1)
        radials_2 = np.arange(0, (2 * np.pi), (2 * np.pi) / aant_2)
        x_coord_1 = RADIUSTWO[0] * np.cos(radials_1)
        y_coord_1 = RADIUSTWO[0] * np.sin(radials_1)
        x_coord_2 = RADIUSTWO[1] * np.cos(radials_2)
        y_coord_2 = RADIUSTWO[1] * np.sin(radials_2)
        x_coord = np.concatenate((x_coord_1, x_coord_2), axis=0)
        y_coord = np.concatenate((y_coord_1, y_coord_2), axis=0)
        cities = np.stack((x_coord, y_coord), axis=-1)
    else:
        # city distribution is random
        cities = np.random.random_sample((AANTSTEDEN, 2))
        x_coord = cities[:, 0]
        y_coord = cities[:, 1]
    return cities, x_coord, y_coord

def calc_squareroot(x_1, x_2, y_1, y_2):
    """Calculates the distance between coordinates (x1, y1) and (x2, y2)"""
    the_distance = sqrt((x_2 - x_1) * (x_2 - x_1) + (y_2 - y_1) * (y_2 - y_1))
    return the_distance

def calc_distances_between_cities(cities):
    """Creates a dictionary with all the distances between every city.

    returns dictionary
      each element is like {(0,1):'distance between city[0] and city[1]'}
    """
    my_dist = {}
    number_of_cities = len(cities)
    for teller1 in range(number_of_cities):
        for teller2 in range(number_of_cities):
            my_dist[(teller1, teller2)] = calc_squareroot(cities[teller1][0],
                                                          cities[teller2][0],
                                                          cities[teller1][1],
                                                          cities[teller2][1])
    return my_dist

def create_random_generation():
    """Creates array of [SIZEPOPULATION, AANTSTEDEN] with random numbers"""
    rkp_generation = np.random.random_sample((SIZEPOPULATION, AANTSTEDEN))
    return rkp_generation

def get_clean_generation(rkp_generation):
    """Converts random numbers to integer city numbers"""
    generation = []
    # Sort every row in array
    sorted_rkp_generation = np.sort(rkp_generation)
    for counter in range(SIZEPOPULATION):
        new_gene = []
        for city in sorted_rkp_generation[counter]:
            new_gene.append(np.where(rkp_generation[counter] == city)[0][0])
        generation.append(new_gene)
    return generation

def calc_individual_distances(total_pop, distance_table):
    """Returns list with all distances of each chromosome

    input:
        total_pop = a RKP list of the cities
        distance_table = dict with distances between cities
    ouput: list with distance of each chromosome"""
    distance_list = []
    for trip in get_clean_generation(total_pop):
        if TRIPSHOULDBELOOP:
            trip_distance = distance_table[(trip[0], trip[-1])]
        else:
            trip_distance = 0
        for city in range(AANTSTEDEN - 1):
            trip_distance += distance_table[(trip[city], trip[city + 1])]
        # now distance_list contains all the distances per individual
        distance_list.append(trip_distance)
    return distance_list

def power(my_list):
    """calculates power of each item in list and returns that list"""
    return [x**FITNESSTOTHEPOWER for x in my_list]

def calc_fitnesses(distance_list):
    """returns accumulated fitness list

    first calculates the percentages of the population_distance, then
    calculates the power and then accumulates these values.
    example: [1, 3, 6] returns [0.1, 0.4, 1.0] for FITNESSTOTHEPOWER=1
    example: [1, 3, 6] returns [0.01, 0.1, 0.46] for FITNESSTOTHEPOWER=2
    """
    fitnesses_list = []
    selection_list = []
    population_distance = sum(distance_list)
    # calc fitness_list as list of fractions of total distance
    for individual_distance in distance_list:
        fitnesses_list.append(individual_distance / population_distance)
    if FITNESSTOTHEPOWER != 1:
        # may be used to increase differences
        temp_list = power(fitnesses_list)
        temp_sum = sum(temp_list)
        fitnesses_list = []
        for ind in temp_list:
            fitnesses_list.append(ind / temp_sum)
    # calculate rolling sum of list
    selection_list = list(accumulate(fitnesses_list))
    # normalize selection_list to values between [0, 1]
    selection_list = [x / selection_list[-1] for x in selection_list]
    return selection_list

def select_two_random(fitness):
    """Selects two random indexes calculated from fitness"""
    selection_list = []
    for _ in range(2):
        selection_value = np.random.random_sample()
        selection_list.append(next(i for i, v in enumerate(fitness) if v >
                                   selection_value))
    return selection_list

def split_two_at_random_gene(individual_a, individual_b):
    """Splits 2 chromosomes at random place and switches halfs"""
    split = np.random.randint(AANTSTEDEN)
    new_a = np.concatenate((individual_a[:split],
                            individual_b[split:]),
                           axis=0)
    new_b = np.concatenate((individual_b[:split],
                            individual_a[split:]),
                           axis=0)
    return new_a, new_b

def split_two_at_two(individual_a, individual_b):
    split = np.sort(np.random.randint(1, AANTSTEDEN, 2))
    split1 = split[0]
    split2 = split[1]
    a_1 = individual_a[:split1]
    a_2 = individual_a[split1:split2]
    a_3 = individual_a[split2:]
    b_1 = individual_b[:split1]
    b_2 = individual_b[split1:split2]
    b_3 = individual_b[split2:]
    new_1 = np.concatenate((a_1, b_2, a_3), axis=0)
    new_2 = np.concatenate((b_1, a_2, b_3), axis=0)
    new_3 = np.concatenate((a_1, a_2, b_3), axis=0)
    new_4 = np.concatenate((b_1, b_2, a_3), axis=0)
    new_5 = np.concatenate((a_1, b_2, b_3), axis=0)
    new_6 = np.concatenate((b_1, a_2, a_3), axis=0)
    return [new_1, new_2, new_3, new_4, new_5, new_6]

def find_shortest_two(population, distances):
    """Finds two shortest distances and returns corresponding chromosomes"""
    if distances[1] < distances[0]:
        first_shortest, second_shortest = population[1], population[0]
        first_index, second_index = 1, 0
    else:
        first_shortest, second_shortest = population[0], population[1]
        first_index, second_index = 0, 1
    for counter in range(len(distances)):
        if (distances[counter] < distances[first_index]) \
            and not np.array_equal(population[counter],
                                   population[first_index]):
            second_shortest = first_shortest
            second_index = first_index
            first_shortest = population[counter]
            first_index = counter
        if (distances[counter] < distances[second_index]) \
            and not np.array_equal(population[counter],
                                   population[first_index]) \
            and not np.array_equal(population[counter],
                                   population[second_index]):
            second_index = counter
            second_shortest = population[counter]
    shortest_two = np.vstack((first_shortest, second_shortest))
    return shortest_two

def get_roulette_generation(population, fitness, pop_size):
    """Select random chromosomes and recombines them. Returns them in a list.

    This works because fiter individuals have a higher chance to be chosen.
    """
    a_new_generation = []
    while len(a_new_generation) < pop_size:
        random_selection = select_two_random(fitness)
        individual_a = population[random_selection[0]]
        individual_b = population[random_selection[1]]
        new_a, new_b = split_two_at_random_gene(individual_a, individual_b)
        a_new_generation.append(new_a)
        a_new_generation.append(new_b)
    a_new_generation = np.vstack((a_new_generation[:]))
    return a_new_generation

def get_eliteist_generation(population, fitness, individual_distance):
    """finds 2 shortest path chromosomes and keeps them first in list
    """
    shortest_two = find_shortest_two(population, individual_distance)
    rest_population = get_roulette_generation(population,
                                              fitness,
                                              SIZEPOPULATION - 2)
    total_pop = np.vstack((shortest_two, rest_population))
    return total_pop

def fill_superras_rand_split(shortest_two):
    """Takes 2 individuals and fills pop with rand splits of these two"""
    a_new_generation = []
    individual_a = shortest_two[0]
    individual_b = shortest_two[1]
    while len(a_new_generation) < (SIZEPOPULATION - 2):
        new_a, new_b = split_two_at_random_gene(individual_a, individual_b)
        a_new_generation.append(new_a)
        a_new_generation.append(new_b)
    return a_new_generation

def fill_superras_two_split(shortest_two):
    a_new_generation = []
    individual_a = shortest_two[0]
    individual_b = shortest_two[1]
    asked_length = SIZEPOPULATION - 2
    while len(a_new_generation) < (asked_length):
        new_ones = split_two_at_two(individual_a, individual_b)
        for one in new_ones:
            a_new_generation.append(one)
    return a_new_generation[:asked_length]

def fill_superras_rand_chosen_genes(shortest_two):
    """for every gene chooses either parrent random"""
#    import pdb; pdb.set_trace()
    individual_a = shortest_two[0]
    individual_b = shortest_two[1]
    rest_pop = []
    while len(rest_pop) < (SIZEPOPULATION - 2):
        rand_choose = np.random.random_sample((1, AANTSTEDEN))
        new_a, new_b = [], []
        for counter in range(AANTSTEDEN):
            if rand_choose[0][counter] < 0.5:
                new_a.append(individual_a[counter])
                new_b.append(individual_b[counter])
            else:
                new_a.append(individual_b[counter])
                new_b.append(individual_a[counter])
        rest_pop.append(new_a)
        rest_pop.append(new_b)
    return rest_pop

def get_superras_generation(population, individual_distance):
    """Finds 2 shortest path chromosomes and uses them to calc rest."""
    shortest_two = find_shortest_two(population, individual_distance)
    if SUPERMETHOD == "onesplit":
        rest_population = fill_superras_rand_split(shortest_two)
    elif SUPERMETHOD == "twosplit":
        rest_population = fill_superras_two_split(shortest_two)
    elif SUPERMETHOD == "random":
        rest_population = fill_superras_rand_chosen_genes(shortest_two)
    else:
        # SUPERMETHOD == "all"
        chance = np.random.random_sample(1)[0]
        if chance < SUPERALLCHANCES[0]:
            rest_population = fill_superras_rand_split(shortest_two)
        elif chance < SUPERALLCHANCES[1]:
            rest_population = fill_superras_two_split(shortest_two)
        else:
            rest_population = fill_superras_rand_chosen_genes(shortest_two)
    total_pop = np.vstack((shortest_two, np.array(rest_population)))
    return total_pop

def get_new_generation(population, fitness, individual_distance):
    """Creates the new generation using different procedures"""
    if SELECTIONCRITERIUM == "eliteist":
        a_new_generation = get_eliteist_generation(population, fitness,
                                                   individual_distance)
    elif SELECTIONCRITERIUM == "superras":
        a_new_generation = get_superras_generation(population,
                                                   individual_distance)
    else:
        # selection is following roulette wheel method
        a_new_generation = get_roulette_generation(population, fitness,
                                                   SIZEPOPULATION)
    return a_new_generation

def mutate(population):
    """Creates a population with mutations """
    mutation_population = population
    number_of_mutations = int(AANTSTEDEN * SIZEPOPULATION * MUTATIONRATE)
    for _ in range(number_of_mutations):
        if SELECTIONCRITERIUM in ("eliteist", "superras"):
            random_chromosome = np.random.randint(2, SIZEPOPULATION)
        else:
            random_chromosome = np.random.randint(SIZEPOPULATION)
        random_gene = np.random.randint(AANTSTEDEN)
        random_gene_value = np.random.random_sample()
        mutation_population[random_chromosome, random_gene] = random_gene_value
    return mutation_population

def main():
    """This is the main program loop"""
    np.random.seed(323456)
    steden, cities_x, cities_y = create_cities()
#    print(steden, file=open("steden-323456.txt", "a"))
    np.random.seed()
    plt.scatter(cities_x, cities_y, s=3*np.pi)
    plt.show()
    distances = calc_distances_between_cities(steden)
    the_population = create_random_generation()
    this_population_distances = calc_individual_distances(the_population,
                                                          distances)
    this_populations_fitness = calc_fitnesses(this_population_distances)
    #calc_information()
    print("random generation. Minimal distance is {} and max is {}".format(
        min(this_population_distances), max(this_population_distances)))
    for count_generation in range(NUMBEROFGENERATIONS):
        new_generation = get_new_generation(the_population,
                                            this_populations_fitness,
                                            this_population_distances)
#        import pdb; pdb.set_trace()
        new_generation = mutate(new_generation)
        the_population = new_generation
        this_population_distances = calc_individual_distances(the_population,
                                                              distances)
        this_populations_fitness = calc_fitnesses(this_population_distances)
        #calc_information()
        print("Generation {} with minimal distance {} and max {}".format(
            count_generation, \
            min(this_population_distances), \
            max(this_population_distances)))
    print_chromosome = get_clean_generation(the_population)[0]
    x_coord = []
    y_coord = []
    for city in print_chromosome:
        x_coord.append(steden[city, 0])
        y_coord.append(steden[city, 1])
    plt.plot(x_coord, y_coord, '-o')
    plt.show()

if __name__ == '__main__':
    main()
