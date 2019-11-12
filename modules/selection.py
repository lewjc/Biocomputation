import random
from classes.individual import Individual

def tournament_selection(population):
    offspring = []
    select_parent = lambda x: random.randrange(x)

    population_size = len(population)
    for i in range (population_size):
        parent_one = population[select_parent(population_size)]
        parent_two = population[select_parent(population_size)]

        parent_one = Individual(_id=parent_one.uid, 
                chromosome=parent_one.chromosome.copy(),
                fitness=parent_one.fitness)
        parent_two = Individual(_id=parent_two.uid,
                chromosome=parent_two.chromosome.copy(),
                fitness=parent_two.fitness)

        if(parent_one.fitness > parent_two.fitness):                
            offspring.insert(i, parent_one)
        elif(parent_one.fitness < parent_two.fitness):
            offspring.insert(i, parent_two)
        else:   
            choice = random.randint(0,1)
            if(choice):
                offspring.insert(i, parent_one)
            else:
                offspring.insert(i, parent_two)

    return offspring