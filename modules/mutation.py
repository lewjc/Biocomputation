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
            possibilities = [0,1,'#']
            if ((idx + 1) % rule_size == 0):
                possibilities = [0,1]                
            possibilities.remove(cell)
            cell = random.choice(possibilities)
            
            individual.chromosome[idx] = cell            


def floating_point_boundary_mutate(individual: Individual,
    mutation_probability: float, rule_size: int):
    idx = -1
    for gene in individual.chromosome:
        idx += 1
        chance = random.uniform(0, 1)
        mutate = (chance <= mutation_probability)
        if(mutate):       
            if ((idx + 1) % rule_size == 0):
                gene ^= 1
                individual.chromosome[idx] = gene
                break            
            operator = -1 if random.randint(0, 1) else 1
            mutation = random.uniform(0.01, 0.12) * operator
            mutate_upper_bound = random.randint(0, 1)
            individual.chromosome[idx] = mutate_bound(mutate_upper_bound, gene, mutation)


def mutate_bound(isUpper, gene, mutation):
    bound = gene[1] if isUpper else gene[0]
    if((bound + mutation) < 0 or (bound + mutation) > 1):
        mutation *= -1
    mutated_boundary = bound + mutation
    new_bounds = [gene[0], mutated_boundary] if isUpper else [mutated_boundary, gene[1]]
    gene = ((min(new_bounds), max(new_bounds)))
    
    return gene