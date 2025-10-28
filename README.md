# Travelling Thief Ant Colony Optimisation (ACO)

This project implements an **Ant Colony Optimisation (ACO)** algorithm to solve the **Travelling Thief Problem (TTP)** — a challenging combination of the Travelling Salesman Problem (TSP) and the Knapsack Problem.

---

## 🧠 What is the Travelling Thief Problem?

The TTP combines two classic problems:
- **TSP:** Find the shortest route that visits every city once.
- **Knapsack:** Choose items with maximum profit without exceeding capacity.

The twist: picking up items makes you heavier, which slows you down.  
The goal is to **maximise total profit while minimising travel time**.

---

## 🐜 How ACO Works

Ant Colony Optimisation is inspired by how real ants find optimal paths using **pheromone trails**.  
In this project, ants explore different city tours and item selections, guided by:
- Pheromone values (collective learning)
- Heuristic information (distance, profit/weight)
- Random exploration

Over many iterations, the colony converges towards a strong solution.
