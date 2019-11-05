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
            mutation = random.uniform(0, 0.1) * operator if random.randint(0, 1) else '#'
            if(isinstance(mutation, str)):
                gene = ((gene[0], mutation) if random.randint(0, 1) else 
                    (mutation, gene[1]))
            else:
                mutate_upper_bound = random.randint(0, 1)
                if(mutate_upper_bound):
                    ub = gene[1]
                    mutated_boundary = (ub + mutation if isinstance(ub, float) else 
                        (random.random() + mutation))
                    new_bounds = [gene[0], mutated_boundary]
                    if(any(isinstance(x, str) for x in new_bounds)):
                        gene = ((new_bounds[0], new_bounds[1]))
                    else:
                        gene = ((min(new_bounds), max(new_bounds)))
                else:
                    ub = gene[0]
                    mutated_boundary = (ub + mutation if isinstance(ub, float) 
                        else random.random() + mutation)
                    new_bounds = [mutated_boundary, gene[1]]
                    if(any(isinstance(x, str) for x in new_bounds)):
                        gene = ((new_bounds[0], new_bounds[1]))
                    else:
                        gene = ((min(new_bounds), max(new_bounds)))
            