import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import clonalg

b_lo, b_up = (-5, 5)
population_size = 100
selection_size = 10
problem_size = 2
random_cells_num = 20
clone_rate = 20
mutation_rate = 0.2
stop_codition = 200
stop = 0

population = clonalg.create_random_cells(population_size, problem_size, b_lo, b_up)
best_affinity_it = []

while stop != stop_codition:
    population_affinity = [(p_i, clonalg.affinity(p_i)) for p_i in population]
    populatin_affinity = sorted(population_affinity, key=lambda x: x[1])

    best_affinity_it.append(populatin_affinity[:5])

    population_select = populatin_affinity[:selection_size]

    population_clones = []
    for p_i in population_select:
        p_i_clones = clonalg.clone(p_i, clone_rate)
        population_clones += p_i_clones

    pop_clones_tmp = []
    for p_i in population_clones:
        ind_tmp = clonalg.hypermutate(p_i, mutation_rate, b_lo, b_up)
        pop_clones_tmp.append(ind_tmp)
    population_clones = pop_clones_tmp
    del pop_clones_tmp

    population = clonalg.select(populatin_affinity, population_clones, population_size)
    population_rand = clonalg.create_random_cells(random_cells_num, problem_size, b_lo, b_up)
    population_rand_affinity = [(p_i, clonalg.affinity(p_i)) for p_i in population_rand]
    population_rand_affinity = sorted(population_rand_affinity, key=lambda x: x[1])
    population = clonalg.replace(population_affinity, population_rand_affinity, population_size)
    population = [p_i[0] for p_i in population]

    stop += 1

best_individual = sorted([(p_i, clonalg.affinity(p_i)) for p_i in population], key=lambda x: x[1])[0]

def plot_function_and_population_3d(population, best_individual):
    def f(x1, x2):
        return x1**2 + x2**2

    x1 = np.linspace(b_lo, b_up, 100)
    x2 = np.linspace(b_lo, b_up, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    population_2d = [(ind[0], ind[1]) for ind in population]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.set_title("Function and Final Population", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_zlabel("$f(x_1, x_2)$", fontsize=12)

    x_pop = [p[0] for p in population_2d]
    y_pop = [p[1] for p in population_2d]
    z_pop = [f(p[0], p[1]) for p in population_2d]
    ax.scatter(x_pop, y_pop, z_pop, c='red', label='Population', s=20)

    best_x, best_y = best_individual[0][0], best_individual[0][1]
    best_z = f(best_x, best_y)
    ax.scatter(best_x, best_y, best_z, c='blue', label='Best Individual', s=50)

    ax.legend()
    plt.show()


plot_function_and_population_3d(population, best_individual)

print("Best individual found:")
print(f"Coordinates: {best_individual[0]}")
print(f"Function value: {best_individual[1]}")
