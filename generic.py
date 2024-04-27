import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

# Function to calculate the Euclidean distance between two points
def distance(city1, city2):
    return np.sqrt((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2)

# Function to calculate the total distance of a given path
def total_distance(path, cities):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(cities[path[i]], cities[path[i + 1]])
    dist += distance(cities[path[-1]], cities[path[0]])  # Closing the loop
    return dist

# Function to create an initial population of paths
def initialize_population(cities, population_size):
    population = []
    for _ in range(population_size):
        path = list(range(len(cities)))
        random.shuffle(path)
        population.append(path)
    return population

# Function to perform tournament selection
def tournament_selection(population, fitnesses, tournament_size):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    return min(selected, key=lambda x: x[1])[0]

# Function to perform crossover (Partially Mapped Crossover - PMX)
def crossover(parent1, parent2):
    size = len(parent1)
    crossover_points = sorted(random.sample(range(size), 2))
    start, end = crossover_points
    
    # Create children by exchanging segments
    child1 = [-1] * size
    child2 = [-1] * size

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # Fill in the missing elements
    def fill_child(child, parent):
        # Find the elements that need to be filled
        parent_elements = set(parent) - set(child[start:end])
        
        # Create a continuous list of indices to fill the child
        fill_indices = [i for i in range(start, size)] + [i for i in range(start)]
        
        # Fill the child with remaining elements
        for idx in fill_indices:
            if child[idx] == -1:
                child[idx] = next(iter(parent_elements))
                parent_elements.remove(child[idx])

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2

# Function to perform mutation (swap two random cities)
def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(path)), 2)
        path[idx1], path[idx2] = path[idx2], path[idx1]

# Main function for the Genetic Algorithm
def genetic_algorithm(cities, population_size=100, generations=500, tournament_size=5, mutation_rate=0.01):
    # Initialize population
    population = initialize_population(cities, population_size)

    for generation in range(generations):
        fitnesses = [total_distance(path, cities) for path in population]

        # Create a new population
        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            child1, child2 = crossover(parent1, parent2)

            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

        if generation % 50 == 0:
            best_path = min(zip(population, fitnesses), key=lambda x: x[1])
            print(f"Generation {generation}: Best distance = {best_path[1]}")

    best_path = min(zip(population, fitnesses), key=lambda x: x[1])

    return best_path[0], best_path[1]

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Solve the Traveling Salesman Problem with a Genetic Algorithm.")
    parser.add_argument("--algorithm", type=str, required=True, help="The algorithm to use (ga)")
    parser.add_argument("--file", type=str, required=True, help="File with city data")

    args = parser.parse_args()

    if args.algorithm.lower() != "ga":
        raise ValueError("Only the Genetic Algorithm (GA) is implemented.")

    # Ensure the file is provided and exists
    if not args.file:
        raise ValueError("File path is required.")

    import os
    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"File '{args.file}' not found.")

    # Load city data
    cities = pd.read_csv(args.file, sep='\s+').values.tolist()

    # Run the Genetic Algorithm
    best_path, best_distance = genetic_algorithm(cities)

    print("Best path:", best_path)
    print("Best distance:", best_distance)

    # Optionally, plot the best path
    latitudes = [cities[i][1] for i in best_path] + [cities[best_path[0]][1]]
    longitudes = [cities[i][2] for i in best_path] + [cities[best_path[0]][2]]

    plt.figure(figsize=(8, 6))
    plt.plot(longitudes, latitudes, "o-", label="Path")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Best TSP Path")
    plt.show()

if __name__ == "__main__":
    main()
