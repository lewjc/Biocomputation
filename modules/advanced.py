def __cull_population(population, percent=20):
    # To sort the list in place...
    population.sort(key=lambda x: x.fitness, reverse=False)
    # To return a new list, use the sorted() built-in function...
    population = sorted(population, key=lambda x: x.fitness, reverse=False)
    population_length = len(population)
    cull_prop = int(population_length * (percent / 100))
    return population[cull_prop]