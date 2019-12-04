import random
from classes.individual import Individual


def bit_flip_mutate(individual: Individual, mutation_probabilty: float):
    idx = -1
    for cell in individual.chromosome:
        idx += 1
        chance = random.uniform(0, 1)
        mutate = (chance <= mutation_probabilty)
        if(mutate):
            cell ^= 1
            individual.chromosome[idx] = cell
        else:
            continue


def rule_based_mutate(individual: Individual,
                      mutation_probability: float, rule_size: int):
    idx = -1
    for cell in individual.chromosome:
        idx += 1
        chance = random.uniform(0, 1)
        mutate = (chance <= mutation_probability)
        if(mutate):
            possibilities = [0, 1, '#']
            if ((idx + 1) % rule_size == 0):
                possibilities = [0, 1]
            possibilities.remove(cell)
            cell = random.choice(possibilities)

            individual.chromosome[idx] = cell


def floating_point_boundary_mutate(individual: Individual,
                                   mutation_probability: float, rule_size: int):
    idx = -1
    for gene in individual.chromosome:
        idx += 1

        bound_idx = 0
        chance = random.uniform(0, 1)
        mutate = (chance <= mutation_probability)

        if (not isinstance(gene, tuple)):
            if(mutate):
                gene ^= 1
                individual.chromosome[idx] = gene
            continue

        bounds = []
        for bound in gene:
            chance = random.uniform(0, 1)
            mutate = (chance <= mutation_probability)
            bounds.append(bound)
            if(mutate):
                operator = -1 if random.randint(0, 1) else 1
                mutation = random.uniform(0.01, 0.15) * operator
                bounds[bound_idx] = mutate_bound(bound, mutation)
            bound_idx += 1

        individual.chromosome[idx] = (min(bounds), max(bounds))


def mutate_bound(bound, mutation):
    if((bound + mutation) < 0 or (bound + mutation) > 1):
        mutation *= -1
    mutated_boundary = bound + mutation
    return mutated_boundary
