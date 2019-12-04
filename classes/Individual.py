class Individual():

    fitness = 0
    chromosome = None
    prediction = 0
    uid = None

    def __init__(self, _id, chromosome=[], fitness=0, prediction=None, ):
        '''
            An individual represents one candidate solution to a problem.

        '''
        self.uid = _id
        self.chromosome = chromosome
        self.fitness = fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __str__(self):
        return 'Individual [ID: {} chromosome: {} fitness: {}]\n'.format(self.uid,
                                                                         self.chromosome, self.fitness)
