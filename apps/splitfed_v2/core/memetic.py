# Import necessary libraries
import random


# result, population = memetic_algorithm(population_size=50, population_length=len(fitness.cluster.clients),
#                                        generations=10, crossover_rate=0.8, mutation_rate=0.3,
#                                        local_search_probability=0.2, fitness_func=fitness.evaluate)
def memetic_algorithm(population_size, population_length, generations, crossover_rate, mutation_rate,
                      local_search_probability, fitness_func):
    """
    the implementation of the memetic algorithm

    Args:
        population_size: how many individual we create
        population_length: how much each individual have objects (clients, participants)
        generations: how many iterations before achieving the best solution
        crossover_rate: between 0-1
        mutation_rate: between 0-1
        local_search_probability:
        fitness_func: fitness function to evaluate each individual,
            better pass it from class to keep the map to the original subjects

    Returns: best solution

    """
    population = initialize_population(population_size, population_length)

    for generation in range(generations):
        offspring = genetic_operators(population, crossover_rate, mutation_rate)

        for i, ind in enumerate(offspring):
            if random.random() < local_search_probability:
                offspring[i] = local_search(ind)

        combined_population = population + offspring
        fitness_scores = [fitness_func(f) for f in combined_population]
        population = wheel_selection(combined_population, fitness_scores, len(combined_population) // 2)
        # population = tournament_selection(combined_population, 2, fitness_func, mapper)
    return get_best_solution(population, fitness_func), population


def initialize_population(population_size, individual_length, min_non_zeros=0.1):
    population = []

    for _ in range(population_size):
        individual = [random.choice([0, 1]) for _ in range(individual_length)]
        non_zeros = sum(individual)

        while non_zeros < individual_length * min_non_zeros:
            index_to_flip = random.randint(0, individual_length - 1)
            if individual[index_to_flip] == 0:
                individual[index_to_flip] = 1
            non_zeros = sum(individual)

        population.append(individual)

    return population


def evaluate_population(population, evaluate_individual):
    fitness_scores = [evaluate_individual(individual) for individual in population]
    return fitness_scores


def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2
    else:
        return parent1, parent2


def mutation(individual, mutation_rate):
    mutated_individual = [bit ^ (random.random() < mutation_rate) for bit in individual]
    return mutated_individual


def genetic_operators(population, crossover_rate, mutation_rate):
    offspring = []

    for i in range(0, len(population), 2):
        parent1 = population[i]
        parent2 = population[i + 1] if i + 1 < len(population) else population[i]

        child1, child2 = crossover(parent1, parent2, crossover_rate)

        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)

        offspring.extend([child1, child2])

    return offspring


def local_search(individual):
    return individual


def tournament_selection(population, tournament_size, evaluate_individual):
    selected_parents = []

    for _ in range(len(population) // 2):
        tournament_candidates = random.sample(population, tournament_size)
        winner = max(tournament_candidates, key=evaluate_individual)
        selected_parents.append(winner)

    return selected_parents


def wheel_selection(population, fitness_scores, num_selections):
    selected_population = []

    for _ in range(num_selections):
        if not population:
            break  # Break if all individuals have been selected

        total_fitness = sum(fitness_scores)
        selection_probabilities = [fitness / total_fitness for fitness in fitness_scores]

        # Spin the wheel
        spin = random.uniform(0, 1)

        # Select individuals based on the wheel
        cumulative_probability = 0
        selected_index = None

        for i, probability in enumerate(selection_probabilities):
            cumulative_probability += probability
            if spin <= cumulative_probability:
                selected_index = i
                break  # Break after selecting one individual

        if selected_index is not None:
            selected_individual = population.pop(selected_index)  # Remove selected individual
            selected_population.append(selected_individual)

    return selected_population


def get_best_solution(population, evaluate_individual):
    return max(population, key=evaluate_individual)
