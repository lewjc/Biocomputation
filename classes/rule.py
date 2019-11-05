class Rule():

    def __init__(self, feature, label, match_count=0):
        self.feature = feature
        self.label = label
        self.match_count = match_count

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
