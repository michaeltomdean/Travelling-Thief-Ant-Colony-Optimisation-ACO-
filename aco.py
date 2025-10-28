"""
Author: Michael Dean
University of Exeter

ACO Implementation of the security van packing problem.

"""

# Library imports
import random
import time
import yaml
import multiprocessing

# Fixed constants
NO_BAGS = 100

"""
Modify the number of CPU cores based on your system. Using multhreading.cpu_count() on intel's big
little architecture will return performance + effiency core count. Code runs significantly better
on performance cores alone! For an i7-14700k used there are 8 performance cores.
"""
NO_CPU_CORES = 8 

def main() -> None:
    """
    Main function
    """
    print("Generating Graphs and Matrices")

    start_time = time.time()
    # Run an ACO test on security van problem
    (_, best_solution) = run_aco_simulation(no_evaluations=25000, no_ants=500, alpha=1, beta=2,
                                                        e=0.95, m=1, smart_start=False, log_all = False, debug=True)
    print(f"Time: {time.time()-start_time}")
    print(best_solution)
    

def run_aco_simulation(no_evaluations: int, no_ants: int, alpha: int, beta: int, m: float, e: float, smart_start: bool,
                       log_all: bool, debug: bool) -> (list, list):
    """
    High-level function to run an ACO solution to the security van packing problem

    :param no_evaluations: The number of fitness evaluations you want the algorithm to perform
    :param no_ants: The number of ants in each tour
    :param alpha: Probability constant
    :param beta: Probability constant
    :param m: Pheromone scaling factor
    :param e: Pheromone decay rate
    :param smart_start: Experimental feature to include a smarter start to the algorithm instead of a random start
    :param log_all: Advanced mode which will log all ants solutions not just the best one for each ant
    :param debug: Print statements on / off
    :return:
    """

    random.seed() # Change seed
    data = import_data()
    graph = generate_bag_graph()
    hm = generate_heuristic_matrix(graph, data)
    phm = generate_pheromone_matrix()

    # Solution storing variables
    all_solutions = [{"solutions": [], "values": []} for i in range(no_ants)]
    ants_best_solution = {"visited bags": [], "value": 0}

    if debug:
        print("Starting evaluations")
        start_time = time.time()
    for evaluations in range(1, (no_evaluations+1)//no_ants): # Convert the number of evaluations into the number of tours required

        # Create a multiprocessing pool to calculate all the ants paths in parallel
        with multiprocessing.Pool(processes=NO_CPU_CORES) as pool: #Numpy seemed to make the multithreading code signifcantly slower... used default python lists & dicts
            arguments = (smart_start, data, graph, hm, phm, alpha, beta)
            tasks = [pool.apply_async(run_singe_ant_sim, args=arguments) for _ in range(no_ants)]
            results = [task.get() for task in tasks] # Await all the results from the pool can't update pheromones untill all ants are done

        # Apply evaporation and then apply pheromone for each ant's tour
        phm = evaporate_pheromone(phm, e)
        for ant_no, ant_result in enumerate(results):
            ants_money = 0
            for no in ant_result:
                ants_money += data["bags"][f"bag_{no}"]["value"]

            # Compare the solution to the best solution all ants has found so far, if its better update it
            if ants_money > ants_best_solution["value"]:
                ants_best_solution["value"] = ants_money
                ants_best_solution["visited bags"] = ant_result

            # Add the solution to the all solution array for debug if log_all is true
            if log_all:
                all_solutions[ant_no]["solutions"].append(ant_result)
                all_solutions[ant_no]["values"].append(ants_money)

            # Apply pheromone proportional to the total money deposited into the van to each edge of the graph visited
            for index, no in enumerate(ant_result):
                if index != len(ant_result) - 2:  # Before the last index break
                    phm[index][
                        ant_result[index + 1] - 1] +=  m * 1 * (ants_money/ants_best_solution["value"]) # Fitness function
                else:
                    break

        if (evaluations%10 == 0) and debug:
            print(f"{(evaluations*no_ants)} - evaluations completed | average time per evaluation - {round((time.time() - start_time)/(evaluations*no_ants), 4)}s")


    return all_solutions if log_all == True else None, ants_best_solution


def run_singe_ant_sim(smart_start, data, graph, heuristic_matrix, pheromone_matrix, alpha, beta) -> list:
    """
    Singular ant pathfinder function, run in parallel during a tour

    :param smart_start: Experiment to include a smarter start to the algorithm instead of a random start
    :param data: YAML data read from BankProblem.txt
    :param graph: Construction graph relation each node to each other node
    :param heuristic_matrix: asd
    :param pheromone_matrix: matrix containing the ant pheromone deposits from each bag to each other bag
    :param alpha: Probability constant
    :param beta: Probability constant
    :return: list of the visited nodes in that ants path
    """

    # Experiment with a "smart start" or simply choose a random bag as a starting point
    if smart_start:
        start_bag = heuristic_matrix[0].index(max(heuristic_matrix[0])) + 1 # The bag with the best money/weight ratio
    else:
        start_bag = 0
        while start_bag == 0:
            start_bag = random.choice(graph[0]) # Random starting position on graph. Ensure it is not a zero (i.e a
                                                                            #row where you've chosen the start point)

    # Starting conditions
    visited_bags = [start_bag]
    current_weight = data["bags"][f"bag_{start_bag}"]["weight"]

    hm = heuristic_matrix
    bust = False  # You go "bust" when your next selected node increases the total weight to more than the van
                      # can cary (like blackjack terminolgy)

    while not bust:
        # Update the bag to current bag you are visiting
        current_bag = visited_bags[-1]

        # Set node to visited
        for i in range(NO_BAGS):
                hm[i][current_bag - 1] = 0

        # Calculate cumulative probability
        cproba = calculate_probability(hm, pheromone_matrix, current_bag - 1, alpha, beta)

        # Choose the next bag to visit based on cumulative probabilities
        next_bag = random.choices(graph[current_bag - 1], cum_weights=cproba)
        next_bag = next_bag[0]  # Convert back to int

        # If total weight is not over the security van limit, continue else you've gone bust don't continue
        if current_weight + data["bags"][f"bag_{next_bag}"]["weight"] < data["security_van_capacity"]:
            visited_bags.append(next_bag)
            current_weight += data["bags"][f"bag_{next_bag}"]["weight"]
        else:
            bust = True

    return visited_bags

def import_data() -> dict:
    """
    Import data from BankProblem.txt
    :return: dictionary containing all data
    """
    with open("BankProblem.txt", "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def generate_bag_graph() -> list:
    """

    :return:
    """
    g = []
    covered_nodes = []
    for x in range(1, NO_BAGS+1):
        new_row = []
        for z in covered_nodes:
            new_row.append(z)
        for y in range(x, NO_BAGS+1):
            new_row.append(y)
        new_row[x-1] = 0
        covered_nodes.append(x)
        g.append(new_row)

    return g

def find_best_result(best_solutions) -> (list, list):
    """
    Find the best result from all of the ants best individual results

    :param best_solutions: The best solutions for each ant in an ACO run
    :return: maximum monetary value in the best solution and the visited nodes as a list
    """
    values = []
    for solution in best_solutions:
        values.append(solution["value"])
    max_value = max(values)
    max_value_index = values.index(max_value)
    solution = best_solutions[max_value_index]["visited bags"]
    return max_value, solution

def generate_heuristic_matrix(g: list, data: dict) -> list:
    """
    Generate a heuristic matrix from the construction graph

    :param g: construction graph
    :param data: data read in from the BankProblem.txt file
    :return: the heurisitic matrix 
    """
    hm = []
    bags = data["bags"]
    for row in g:
        new_row = []
        for node in row:
            if node != 0:
                weight = bags[f"bag_{node}"]["weight"]
                money = bags[f"bag_{node}"]["value"]
                new_row.append(money/weight)
            else:
                new_row.append(0)
        hm.append(new_row)
    return hm

def generate_pheromone_matrix():
    """
    Matrix representing randomly distrubuted pheromones from 0-1 on each edge of the graph 
    :return: the created pheromone matrix with randomly distributed pheromone
    """
    return [[random.uniform(0, 1) for _ in range(NO_BAGS)] for _ in range(NO_BAGS)]

def evaporate_pheromone(pm, e):
    """
    Evaporate the pheromones in the pheromone matrix during each tour

    :param pm: Pheremone matrix
    :param rho: Evaporation parameter
    :return: New pheremone matrix
    """
    return [[(1-e)*pm[x][y] for x in range(NO_BAGS)] for y in range(NO_BAGS)]

def calculate_probability(hm, pm, current_node_index, alpha, beta) -> list:
    """
    Calculate the cumulative probability between traveling from the current node to another node

    :param hm: Heuristic matrix
    :param pm: Pheromone matrix
    :param current_node_index: Current node you are at
    :param alpha: Probability constant
    :param beta: Probability constant
    :return: Cumulative probability list for traveling between each other node
    """
    numa = []
    for next_node in range(NO_BAGS):
        numa.append((pm[current_node_index][next_node]**alpha) * (hm[current_node_index][next_node]**beta))

    denom = sum(numa)

    # Calculate the probability between the current bags and all other bags
    proba = [x/denom for x in numa]

    # Calculate the cumulative probability
    total = 0
    cproba = []
    for probability in proba:
        cproba.append(probability+total)
        total += probability
    return cproba


if __name__ == "__main__":
    main()