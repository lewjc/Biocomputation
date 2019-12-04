import plotly.graph_objects as go
from classes.floating_point_data import FloatingPointData
from classes.binary_data import BinaryData


def draw_graph(generation_info, epochs, crossover_probability,
               mutation_probabilty, rule_count, population_size, max_test_fitness=None):
    def flatten_point(l, i): return [sublist[i] for sublist in l]

    if (len(generation_info[0]) == 3):
        generation = flatten_point(generation_info, 0)
        averages = flatten_point(generation_info, 1)
        best = flatten_point(generation_info, 2)
    else:
        generation = flatten_point(generation_info, 0)
        averages = flatten_point(generation_info, 1)
        best = flatten_point(generation_info, 2)
        average_test = flatten_point(generation_info, 3)
        best_test = flatten_point(generation_info, 4)

    figure = go.Figure()

    figure.add_trace(go.Scatter(y=best, x=generation, name="Best Indvidual Fitness",
                                line=dict(color='firebrick', width=4)))
    figure.add_trace(go.Scatter(y=averages, x=generation, name="Average Population Fitness",
                                line=dict(color='royalblue', width=4, dash='dot')))
    figure.add_trace(go.Scatter(y=average_test, x=generation, name="Average Test Fitness",
                                line=dict(color='green', width=4, dash='dot')))
    figure.add_trace(go.Scatter(y=best_test, x=generation, name="Best Test Fitness",
                                line=dict(color='orange', width=4)))
    if(max_test_fitness):
        figure.add_trace(go.Scatter(y=([max_test_fitness] * epochs), x=generation, name="Max Test Fitness",
                                    line=dict(color='black', width=4)))

    title = 'Epoch: {}, Crossover: {}, Mutation: {}, Rule Count: {}, Population: {} '.format(
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


def initialise_floating_point_data(dataset):
    data_props = dataset[0].split(" ")
    amount_of_points = len(data_props) - 1
    prediction_size = len(data_props[-1])
    data = []
    for row in dataset:
        data_props = row.split(" ")
        features = list(map(lambda x: float(x), data_props[:-1]))
        data.append(FloatingPointData(features, int(data_props[-1])))

    return(data, amount_of_points, prediction_size)


def open_and_sanitize_data(data_set_location):
    with open(data_set_location, mode="r") as f:
        data = f.readlines()
        for i in range(len(data)):
            line = data[i]
            data[i] = line.strip("\n")
        return data
