class Rule():

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    @staticmethod
    def generate_rules_from_chromosome(chromosome, feature_size,
        rule_size):   

        rules = []  
        
        for i in range(0, len(chromosome), rule_size):
            rule_stop = (i + feature_size)
            feature = chromosome[i:rule_stop]
            prediction = chromosome[rule_stop: rule_stop + 1][0]
            rules.append(Rule(feature=feature, label=prediction))

        return rules
