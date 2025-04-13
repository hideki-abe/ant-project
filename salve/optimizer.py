import numpy as np
import pickle
import os
from multiprocessing import Pool, cpu_count

from salve.fitness import evaluate_ant_with_sine_waves

checkpoint_dir = "checkpoint"
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")

def save_checkpoint(population, generation):
    with open(checkpoint_file, "wb") as f:
        pickle.dump({
            "population": population,
            "generation": generation
        }, f)

def load_checkpoint():
    with open(checkpoint_file, "rb") as f:
        data = pickle.load(f)
    return data["population"], data["generation"]

def genetic_algorithm(
    pop_size=80,
    generations=30,
    mutation_rate=0.3,
    tournament_size=2,
    elitism_rate=0.1  # Novo parâmetro para elitismo
):
    num_joints = 8

    def new_individual():
        freqs = np.random.uniform(0.5, 3.0, num_joints)
        amps = np.random.uniform(0.2, 1.0, num_joints)
        phases = np.random.uniform(0, 2 * np.pi, num_joints)
        return np.concatenate((freqs, amps, phases))

    def crossover(parent1, parent2):
        alpha = 0.5
        return alpha * parent1 + (1 - alpha) * parent2

    def mutation(individual, mutation_rate):
        if np.random.rand() < mutation_rate:
            new_ind = individual.copy()
            indices = np.random.randint(0, 24, size=12)
            for i in indices:
                new_ind[i] += np.random.normal(0, 0.5)
            return new_ind
        return individual

    def tournament_selection(population, fitness):
        selected = np.random.choice(len(population), tournament_size)
        best = selected[np.argmax([fitness[i] for i in selected])]
        return population[best]

    # Carregar checkpoint se existir
    if os.path.exists(checkpoint_file):
        population, start_gen = load_checkpoint()
        print(f"🔁 Retomando da geração {start_gen}")
    else:
        population = [new_individual() for _ in range(pop_size)]
        start_gen = 0

    print(f"Iniciando o cálculo das gerações: Pop: {pop_size}, Gerações: {generations}, Mutação: {mutation_rate}")

    for generation in range(start_gen, generations):
        # Calcular o fitness da população
        with Pool(cpu_count()) as pool:
            fitness = pool.map(evaluate_ant_with_sine_waves, population)

        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        print(f"🔄 Geração {generation} | ✅ Melhor: {best_fitness:.2f} | Média: {mean_fitness:.2f}")

        # Elitismo: preservar os melhores indivíduos
        elite_size = int(pop_size * elitism_rate)
        elite_indices = np.argsort(fitness)[-elite_size:]
        elites = [population[i] for i in elite_indices]

        # Nova população com elites
        new_population = elites.copy()

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            child = crossover(p1, p2)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Salvar checkpoint
        save_checkpoint(population, generation)

        with Pool(cpu_count()) as pool:
            final_fitness = pool.map(evaluate_ant_with_sine_waves, population)

    # Melhor solução
    final_fitness = [evaluate_ant_with_sine_waves(ind) for ind in population]
    best = np.argmax(final_fitness)
    best_individual = population[best]
    print("🏆 Melhor recompensa final:", final_fitness[best])

    np.save("best_individual.npy", best_individual)

    return best_individual

if __name__ == "__main__":
    best = genetic_algorithm()
    evaluate_ant_with_sine_waves(best, render=True, record_video=True)
