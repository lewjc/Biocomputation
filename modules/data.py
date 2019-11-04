import plotly.graph_objects as go
from classes.binary_data import BinaryData

def draw_graph(generation_info, epochs, crossover_probability,
        mutation_probabilty, rule_count, population_size):
    flatten_point = lambda l, i: [sublist[i] for sublist in l]

    generation = flatten_point(generation_info, 0)
    averages = flatten_point(generation_info, 1)
    best = flatten_point(generation_info, 2)

    figure = go.Figure()

    figure.add_trace(go.Scatter(y=best, x=generation, name="Best Indvidual Fitness",
        line=dict(color='firebrick', width=4)))
    figure.add_trace(go.Scatter(y=averages, x=generation, name="Average Population Fitness",
        line=dict(color='royalblue', width=4, dash='dot')))
    title = 'Epoch: {}, Crossover: {}, Mutation: {}\nRule Count: {}, Population: {} '.format(
        epochs, crossover_probability, mutation_probabilty,
        rule_count, population_size)

    figure.update_layout(title=title,
                xaxis_title='Generation',
                yaxis_title='Fitness')
    
    figure.show()

def initialise_data(dataset):
    data_props = dataset[0].split(" ")
    feature_size = len(data_props[0])
    prediction_size = len(data_props[1])
    data = [] 
    for row in dataset:
        data.append(BinaryData(*row.split(" ")))
    
    return(data, feature_size, prediction_size)

def open_and_sanitize_data(data_set_location):
    with open(data_set_location, mode="r") as f:
        data = f.readlines()
        for i in range (len(data)):
            line = data[i]
            data[i] = line.strip("\n")
        return data
