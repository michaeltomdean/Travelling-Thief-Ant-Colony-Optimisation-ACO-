import json
import time

from aco import import_data, generate_bag_graph, generate_heuristic_matrix, generate_pheromone_matrix, run_aco_simulation

def main() -> None:
    """
    Main function
    """

    # print("Starting ant sweep")
    #ant_sweep()
    # print("Finishing ant sweep")

    # print("Starting rho sweep")
    #rho_sweep()
    # print("Finishing rho sweep")

    #q_sweep()

    alpha_and_beta_sweep()

def ant_sweep():

    # Do a sweep of number of ants from one to 1000
    results = {}
    no_ants = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for x in no_ants:
        start_time = time.time()
        print(f"Ant {x}")
        best_solutions_to_avg = []
        for _ in range(25):
            (all_solutions, best_solution) = run_aco_simulation(no_evaluations=10000, no_ants=x, alpha=1, beta=2,
                                                            e=0.5, m=1, smart_start=False, log_all = False, debug=False)


            best_solutions_to_avg.append(best_solution["value"])
        print(f"Time to calc ant {x}: {round((time.time() - start_time))/60, 4}m")
        results[f"{x}"] = best_solutions_to_avg

    with open("ant_sweep2.json", "w") as f:
        json.dump(results, f)

def rho_sweep():
    
    # Do a sweep of rho from 0.5 to 0.95
    results = {}
    for x in linspace(0.5, 0.95, 100):
        start_time = time.time()
        print(f"rho {x}")
        best_solutions_to_avg = []
        for _ in range(25):
            (all_solutions, best_solution) = run_aco_simulation(no_evaluations=10000, no_ants=100, alpha=1, beta=2,
                                                            e=x, m=1, smart_start=False, log_all = False, debug=False)


            best_solutions_to_avg.append(best_solution["value"])
        
        print(f"Time per rho {x}: {round((time.time() - start_time))/60, 4}m")
        results[f"{x}"] = best_solutions_to_avg

    with open("rho_sweep2.json", "w") as f:
        json.dump(results, f)


def q_sweep():

    
    # Do a sweep of rho from 0.5 to 0.95
    results = {}
    q = [10**-5, 10**-4.5, 10**-4, 10**-3.5, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1, 10**-0.5, 10**0, 10**0.5, 10**1, 10**1.5, 
         10**2, 10**2.5, 10**3, 10**3.5, 10**4, 10**4.5, 10**5]
    for x in q:
        start_time = time.time()
        print(f"q {x}")
        best_solutions_to_avg = []
        for _ in range(25):
            (all_solutions, best_solution) = run_aco_simulation(no_evaluations=10000, no_ants=100, alpha=1, beta=2,
                                                            e=0.5, m=x, smart_start=False, log_all = False, debug=False)


            best_solutions_to_avg.append(best_solution["value"])
        
        print(f"Time per q {x}: {round((time.time() - start_time)/60, 4)}m")
        results[f"{x}"] = best_solutions_to_avg

    with open("q_sweep2.json", "w") as f:
        json.dump(results, f)


def trial_heatmap():
    results = {}
    for x in range(25):
        (all_solutions, best_solution) = run_aco_simulation(no_evaluations=10000, no_ants=100, alpha=1, beta=2,
                                                    e=0.5, m=1, smart_start=False, log_all = True, debug=False)
        results[f"{x}"]= all_solutions

    with open("trial_heatmap.json", "w") as f:
        json.dump(results, f)

def alpha_and_beta_sweep():
    # Alpha sweep
    results = {f"{a}": [] for a in range(0, 26)}
    for a in range(0, 26):
        for trial in range(25):
            (all_solutions, best_solution) = run_aco_simulation(no_evaluations=25000, no_ants=500, alpha=a, beta=2,
                                                    e=0.5, m=1, smart_start=False, log_all = True, debug=False)

            results[f"{a}"].append(best_solution["value"])
            print("alpha: ", a)

    with open("alpha_sweep.json", "w") as f:
        json.dump(results, f)

    results = {f"{b}": [] for b in range(0, 26)}
    for b in range(0, 26):
        for trial in range(25):
            (all_solutions, best_solution) = run_aco_simulation(no_evaluations=25000, no_ants=500, alpha=1, beta=b,
                                                                e=0.5, m=1, smart_start=False, log_all=True,
                                                                debug=False)

            results[f"{b}"].append(best_solution["value"])
            print("beta: ", b)

    with open("beta_sweep.json", "w") as f:
        json.dump(results, f)


    
# Use own linspace so code can be run on my Windows 11 ARM laptop where not all pip packages are supported
def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i

if __name__ == "__main__":
    main()

