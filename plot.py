import json
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import networkx as nx

from aco import import_data

def main():

    with open("ant_sweep.json") as f:
        data = json.load(f)

    x = data.keys()
    no_ants = range(1, 101)
    initial_averaged = {f"{x}": 0 for x in no_ants}
    initial_min_max = {f"{x}": {"min": 50000, "max": 0} for x in no_ants}
    average, min, max = average_min_max_old(data, initial_averaged, initial_min_max)
    fig, ax = plt.subplots()
    plt.title("Fitness vs Number of Ants, Evaluations=1000, Trials=25, e=0.5, m=1")
    ax.set_xlabel("Number of Ants")
    ax.set_ylabel("Fitness (£)")
    ax.set_xticks([0, 24, 49, 74, 99])
    ax.plot(x, average, label="Average Fitness")
    ax.plot(x, min, linestyle = 'dashed', label="Minimum Fitness ")
    ax.plot(x, max, linestyle = 'dashed', label="Maximum Fitness")
    plt.legend()
    plt.show()

    with open("ant_sweep2.json", "r") as f:
        data = json.load(f)

    x = data.keys()
    no_ants = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    initial_averaged = {f"{x}": 0 for x in no_ants}
    initial_min_max = {f"{x}": {"min": 50000, "max": 0} for x in no_ants}
    average, min, max = average_min_max(data, initial_averaged, initial_min_max)
    fig, ax = plt.subplots()
    plt.title("Fitness vs Number of Ants, Evaluations=10000, Trials=25, e=0.5, m=1")
    ax.set_xlabel("Number of Ants")
    ax.set_ylabel("Fitness (£)")
    ax.plot(x, average, label="Average Fitness")
    ax.plot(x, min, linestyle = 'dashed', label="Minimum Fitness ")
    ax.plot(x, max, linestyle = 'dashed', label="Maximum Fitness")
    plt.legend()
    plt.show()

    with open("rho_sweep2.json", "r") as f:
        data = json.load(f)

    x = data.keys()
    initial_averaged = {f"{x}": 0 for x in linspace(0.5, 0.95, 100)}
    initial_min_max = {f"{x}": {"min": 50000, "max": 0} for x in linspace(0.5, 0.95, 100)}
    average, min, max = average_min_max(data, initial_averaged, initial_min_max)
    fig, ax = plt.subplots()
    plt.title("Fitness vs Pheromone Evaporation (e) , Evaluations=1000, Number of Ants=100, Trials=25, m=1")
    ax.set_xlabel("Pheromone Evaporation (e)")
    ax.set_ylabel("Fitness (£)")
    ax.set_xticks([0, 55, 99])
    ax.plot(x, average, label="Average Fitness")
    ax.plot(x, min, linestyle = 'dashed', label="Minimum Fitness ")
    ax.plot(x, max, linestyle = 'dashed', label="Maximum Fitness")
    plt.legend()
    plt.show()


    with open("q_sweep2.json", "r") as f:
        data = json.load(f)

    x = data.keys()
    q = [10**-5, 10**-4.5, 10**-4, 10**-3.5, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1, 10**-0.5, 10**0, 10**0.5, 10**1, 10**1.5, 
        10**2, 10**2.5, 10**3, 10**3.5, 10**4, 10**4.5, 10**5]
    initial_averaged = {f"{x}": 0 for x in q}
    initial_min_max = {f"{x}": {"min": 50000, "max": 0} for x in q}
    average, min, max = average_min_max(data, initial_averaged, initial_min_max)
    fig, ax = plt.subplots()
    plt.title("Fitness Pheromone Scaler (m) , Evaluations=10000, Number of Ants=100, Trials=25, e=0.5")
    ax.set_xlabel("Pheromone Sacaler (m)")
    ax.set_xticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e-0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5"])
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_ylabel("Fitness (£)")
    ax.plot(x, average, label="Average Fitness")
    ax.plot(x, min, linestyle = 'dashed', label="Minimum Fitness ")
    ax.plot(x, max, linestyle = 'dashed', label="Maximum Fitness")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()

    bag_data = import_data()
    bags = bag_data["bags"]
    money_to_weight_node_ratio = [[bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(1, 11)], [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(11, 21)], 
    [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(21, 31)], [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(31, 41)], 
    [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(41, 51)], [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(51, 61)], 
    [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(61, 71)], [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(71, 81)], 
    [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(81, 91)], [bags[f"bag_{i}"]["value"]/bags[f"bag_{i}"]["weight"] for i in range(91, 101)]]

    img = ax.imshow(money_to_weight_node_ratio, cmap="magma", norm=LogNorm())
    fig.colorbar(img)
    ax.set_yticks(range(0, 10))
    ax.set_yticklabels(["bags 1-10", "bags 11-20", "bags 21-30", "bags 31-40", "bags 41-50", "bags 51-60", "bags 61-70", "bags 71-80", "bags 81-90", "bags 91-100"])
    ax.set_xticks(range(0, 10))
    ax.set_xticklabels(str(i) for i in range(1, 11))
    plt.title("Money/weight ratio for 100 bags in security van problem")
    plt.show()

    with open("trial_heatmap.json", "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots()
    number_of_times_visited = {f"{x}":0 for x in range(1, 101)}
    for each_trial in data.keys():
        nodes_visited = data[each_trial]
        for each_ant in nodes_visited:
            each_ants_solution = each_ant["solutions"]
            for each_tour in each_ants_solution:
                for each_node in each_tour:
                    number_of_times_visited[str(each_node)] += 1
    
    # Convert to img format
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    row5 = []
    row6 = []
    row7 = []
    row8 = []
    row9 = []
    row10 = []
    for key in number_of_times_visited.keys():
        ekey = int(key)
        if 0 < ekey < 11:
            row1.append(number_of_times_visited[key])
        elif 11 <= ekey < 21:
            row2.append(number_of_times_visited[key])
        elif 21 <= ekey < 31:
            row3.append(number_of_times_visited[key])
        elif 31 <= ekey < 41:
            row4.append(number_of_times_visited[key])
        elif 41 <= ekey < 51:
            row5.append(number_of_times_visited[key])
        elif 51 <= ekey < 61:
            row6.append(number_of_times_visited[key])
        elif 61 <= ekey < 71:
            row7.append(number_of_times_visited[key])
        elif 71 <= ekey < 81:
            row8.append(number_of_times_visited[key])
        elif 81 <= ekey < 91:
            row9.append(number_of_times_visited[key])
        elif 91 <= ekey < 101:
            row10.append(number_of_times_visited[key])

    img = [row1, row2, row3, row4, row5, row6, row7, row8, row9, row10]

    fig, ax = plt.subplots()
    img = ax.imshow(img, cmap="magma", norm=LogNorm())
    fig.colorbar(img)
    ax.set_yticks(range(0, 10))
    ax.set_yticklabels(["bags 1-10", "bags 11-20", "bags 21-30", "bags 31-40", "bags 41-50", "bags 51-60", "bags 61-70", "bags 71-80", "bags 81-90", "bags 91-100"])
    ax.set_xticks(range(0, 10))
    ax.set_xticklabels(str(i) for i in range(1, 11))
    plt.title("Number of node visits by all ants evaluations=10000, number of ants=100, trials=25 e=0.5, m=1")
    plt.show()


        






def average_min_max(data, averaged_data, min_max):
    for key, value in data.items():
        for each_result in value:
            averaged_data[key] += each_result
            if each_result > min_max[key]["max"]:
                min_max[key]["max"] = each_result
            if each_result < min_max[key]["min"]:
                min_max[key]["min"] = each_result

        average = []
        for key in averaged_data.keys():
            print(averaged_data[key])
            average.append(averaged_data[key] / 25)

        min = []
        max = []
        for key in min_max.keys():
            min.append(min_max[key]["min"])
            max.append(min_max[key]["max"])

    return average, min, max

def average_min_max_old(data, averaged_data, min_max):
    for key, value in data.items():
        for each_result in value:
            averaged_data[key] += each_result["value"]
            if each_result["value"] > min_max[key]["max"]:
                min_max[key]["max"] = each_result["value"]
            if each_result["value"] < min_max[key]["min"]:
                min_max[key]["min"] = each_result["value"]

    average = []
    for key in averaged_data.keys():
        print(averaged_data[key])
        average.append(averaged_data[key]/25)

    min = []
    max = []
    for key in min_max.keys():
        min.append(min_max[key]["min"])
        max.append(min_max[key]["max"])

    return average, min, max


def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i

def plot_construction_graph():
    num_bags = 100
    graph = nx.complete_graph(num_bags)
    plt.figure(figsize=(6, 6))
    plt.title("Security Van Problem Construction Graph")
    pos = nx.spring_layout(graph, seed=42) 
    nx.draw(graph, pos, node_size=20, node_color="blue", edge_color="gray", with_labels=False)
    plt.show()

if __name__ == "__main__":
    plot_construction_graph()
    main()