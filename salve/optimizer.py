import numpy as np
import pickle
import os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt  # Adicionado para a função de plotagem

from fitness import evaluate_ant_with_sine_waves

checkpoint_dir = "checkpoint"
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")


# Função para salvar o progresso do algoritmo
def save_checkpoint(population, generation):
    with open(checkpoint_file, "wb") as f:
        pickle.dump({
            "population": population,
            "generation": generation
        }, f)


# Função para carregar o progresso salvo
def load_checkpoint():
    with open(checkpoint_file, "rb") as f:
        data = pickle.load(f)
    return data["population"], data["generation"]


# Função para plotar as ondas senoidais
def plot_sine_waves(params, timesteps=1000, separate_plots=False):
    """
    Plot the sine waves used for controlling the ant

    Args:
        params (list): List of 24 parameters (8 frequencies, 8 amplitudes, and 8 phase shifts)
        timesteps (int): Number of timesteps to plot
        separate_plots (bool): If True, plot each sine wave in a separate subplot
    """
    # Extract frequencies, amplitudes, and phase shifts
    frequencies = params[:8]
    amplitudes = params[8:16]
    phase_shifts = params[16:]

    # Ensure amplitudes are between 0 and 1
    amplitudes = np.clip(amplitudes, 0, 1)

    # Create time array with the same time_factor used in evaluate_ant_with_sine_waves
    time = np.arange(timesteps) / 10.0  # Scale time for reasonable frequencies

    if separate_plots:
        # Create a figure with 8 subplots (2x4 grid)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
        axes = axes.flatten()

        # Plot each sine wave in its own subplot
        for i in range(8):
            actions = amplitudes[i] * np.sin(frequencies[i] * time + phase_shifts[i])
            axes[i].plot(time, actions)
            axes[i].set_title(f"Joint {i + 1}: f={frequencies[i]:.2f}, A={amplitudes[i]:.2f}, φ={phase_shifts[i]:.2f}")
            axes[i].grid(True)

        # Set common labels with proper padding
        fig.text(0.5, 0.01, "Time (scaled by factor 1/20)", ha='center')
        fig.text(0.04, 0.5, "Joint Action", va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout to leave room for labels
        plt.savefig("sine_waves_individual_plots.png")
    else:
        # Create a single figure with all sine waves
        plt.figure(figsize=(12, 8))

        # Plot each sine wave
        for i in range(8):
            actions = amplitudes[i] * np.sin(frequencies[i] * time + phase_shifts[i])
            plt.plot(time, actions,
                     label=f"Joint {i + 1}: f={frequencies[i]:.2f}, A={amplitudes[i]:.2f}, φ={phase_shifts[i]:.2f}")

        plt.title("Sine Wave Control Signals for Ant Joints")
        plt.xlabel("Time (scaled by factor 1/20)")
        plt.ylabel("Joint Action")
        plt.legend()
        plt.grid(True)
        plt.savefig("sine_waves_plot.png")


# Algoritmo genético principal
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

        # Visualizar o melhor indivíduo da geração a cada 5 gerações
        if generation % 5 == 0:
            best_gen_individual = elites[-1]
            print(f"📊 Plotando geração {generation}...")
            plot_sine_waves(best_gen_individual, timesteps=1000, separate_plots=False)

        # Salvar checkpoint
        save_checkpoint(population, generation)

    # Melhor solução
    final_fitness = [evaluate_ant_with_sine_waves(ind) for ind in population]
    best = np.argmax(final_fitness)
    best_individual = population[best]
    print("🏆 Melhor recompensa final:", final_fitness[best])

    # Plotar a melhor solução final
    print("📊 Gerando plot das ondas senoidais do melhor indivíduo...")
    plot_sine_waves(best_individual, timesteps=1000, separate_plots=True)

    np.save("best_individual.npy", best_individual)

    return best_individual


# Executar o algoritmo genético
if __name__ == "__main__":
    best = genetic_algorithm()
    evaluate_ant_with_sine_waves(best, render=True, record_video=True)
