import argparse
from math import sin, cos, sqrt, atan2, radians
import random

EARTH_RADIUS = 6371  # Radius of the Earth in kilometers

class TSP:
    def __init__(self, city_file):
        self.cities = self.read_cities(city_file)
        self.num_cities = len(self.cities)
        self.distances = self.calculate_distances()

    def read_cities(self, city_file):
        cities = {}
        with open(city_file, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split()
                city, lat, lon = parts
                cities[city] = (float(lat), float(lon))
        return cities

    def calculate_distances(self):
        distances = {}
        for city1 in self.cities:
            distances[city1] = {}
            for city2 in self.cities:
                distances[city1][city2] = self.distance(self.cities[city1], self.cities[city2])
        return distances

    def distance(self, city1, city2):
        """
        Calculates the distance between two cities using the Haversine formula.
        """
        lat1, lon1 = radians(city1[0]), radians(city1[1])
        lat2, lon2 = radians(city2[0]), radians(city2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = EARTH_RADIUS * c
        return distance

    def total_distance(self, path):
        total = 0
        for i in range(self.num_cities - 1):
            total += self.distances[path[i]][path[i+1]]
        total += self.distances[path[-1]][path[0]]
        return total

    def hill_climbing(self, initial_path, max_iter=10000):
        current_path = initial_path[:]
        current_distance = self.total_distance(current_path)
        for _ in range(max_iter):
            neighbor = self.get_neighbor(current_path)
            neighbor_distance = self.total_distance(neighbor)
            if neighbor_distance < current_distance:
                current_path = neighbor
                current_distance = neighbor_distance
        return current_path, current_distance

    def get_neighbor(self, path):
        i, j = random.sample(range(self.num_cities), 2)
        neighbor = path[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def random_path(self):
        path = list(self.cities.keys())
        random.shuffle(path)
        return path


def main():
    parser = argparse.ArgumentParser(description='Solve the TSP using hill_climbing algorithm')
    parser.add_argument('--algorithm', choices=['hill_climbing'], required=True, help='Algorithm to use')
    parser.add_argument('--file', required=True, help='Input file containing the city list')
    args = parser.parse_args()

    tsp_solver = TSP(args.file)

    if args.algorithm == 'hill_climbing':
        initial_path = tsp_solver.random_path()
        solution, distance = tsp_solver.hill_climbing(initial_path)

    print('Best route:', solution)
    print('Total Distance:', distance)


if __name__ == '__main__':
    main()
