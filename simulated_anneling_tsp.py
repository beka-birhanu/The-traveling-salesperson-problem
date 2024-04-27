import argparse
import random
import math

city_locations = {}

# Function to extract city data from the file
def extract_city_data(file_path):
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        
        for entry in file:
            city_name, lat, long = entry.strip().split()
            lat, long = float(lat), float(long)
            city_locations[city_name] = (lat, long)
    
    return city_locations

# Function to compute distance between two cities
def compute_distance(city_a, city_b, city_locations):
    lat1, long1 = city_locations[city_a]
    lat2, long2 = city_locations[city_b]
    lat_diff_squared = (lat2 - lat1) ** 2
    long_diff_squared = (long2 - long1) ** 2
    return math.sqrt(lat_diff_squared + long_diff_squared)

# Function to create a distance matrix
def create_distance_matrix(city_coordinates):
    city_names = list(city_coordinates.keys())
    num_cities = len(city_names)
    distance_matrix = [[0] * num_cities for _ in range(num_cities)]
    
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i][j] = compute_distance(city_names[i], city_names[j], city_coordinates)
            
    return distance_matrix

# Function to compute path distance
def compute_path_distance(path, distance_matrix):
    total_distance = 0.0
    num_cities = len(path)
    
    for i in range(num_cities - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    total_distance += distance_matrix[path[-1]][path[0]]
    
    return total_distance

# Function to generate a random permutation of cities
def generate_random_permutation(n):
    numbers = list(range(n))
    random.shuffle(numbers)
    return numbers

# Function to format path
def format_path(city_indices, city_coordinates):
    city_names = list(city_coordinates.keys())
    path_names = [city_names[index] for index in city_indices]
    return " --> ".join(path_names)

# Function to handle temperature decrease
def temperature_decrease(t, initial_temperature, alpha):
    return initial_temperature * math.exp(-alpha * t)

# Simulated annealing algorithm for TSP
def simulated_annealing(file_name):
    cities_graph = extract_city_data(file_name)
    distance_matrix = create_distance_matrix(cities_graph)
    n = len(distance_matrix)
    initial_solution = generate_random_permutation(n)
    current_path = initial_solution.copy()
    best_path = initial_solution.copy()

    current_path_distance = compute_path_distance(current_path, distance_matrix)
    best_path_distance = current_path_distance
    temperature = 1000
    alpha = 0.01
    t = 0

    while temperature > 0.1:
        new_path = current_path.copy()
        new_city1, new_city2 = random.sample(range(len(new_path)), 2)
        new_path[new_city1], new_path[new_city2] = new_path[new_city2], new_path[new_city1]
        new_path_distance = compute_path_distance(new_path, distance_matrix)
        delta_E = new_path_distance - current_path_distance

        if new_path_distance < current_path_distance:
            current_path = new_path.copy()
            current_path_distance = new_path_distance
            if new_path_distance < best_path_distance:
                best_path = current_path.copy()
                best_path_distance = new_path_distance
        else:
            probability = math.exp(delta_E / temperature)
            if random.random() < probability:
                current_path = new_path.copy()
                current_path_distance = new_path_distance
        
        temperature = temperature_decrease(t + 1, temperature, alpha)
        
    formatted_path = format_path(best_path, cities_graph)
    total_distance = compute_path_distance(best_path, distance_matrix)
    return f"Best Path: {formatted_path}\nDistance: {total_distance}"

# Main function to handle command-line arguments and run the chosen algorithm
def main():
    parser = argparse.ArgumentParser(description='Solve the TSP using the specified algorithm.')

    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to use (ga for genetic algorithm, sa for simulated annealing).')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file containing city data.')

    args = parser.parse_args()
    
    # Run the simulated annealing algorithm
    result = simulated_annealing(args.file)
    
    # Print the result
    print(result)

# Run the main function if the script is executed as a script
if __name__ == "__main__":
    main()
