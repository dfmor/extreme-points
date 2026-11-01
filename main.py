import pickle
import re
import time
import numpy as np

from auxiliary_functions import *
from eliminate import eliminate, optimal_vertex
from calculate_bounds import minimize_class

START_CLASS = 18
NUMBER_OF_CLASSES = 20

N_VARIABLES = 7

MUTUAL_INFORMATION_VARIABLES = [(2,),(3,),(2,3)]
CLASS_MATRIX_FILE = "class_matrix.pkl"

MARGINAL_SCENARIO_SELECTION = np.array([[0,1,4],[0,1,5]])
MARGINAL_SCENARIO_LAMBDA = np.array([[0,1,2,3,4,5]])
MARGINAL_SCENARIO_FINAL = np.array([[0,1,2,3,4],[0,1,2,3,5]])
NAMES = ['X0','X1', 'A', 'An', 'B0', 'B1']

BASELINE = 5000
INCREASE = 100

N_EPS = 1
N_BISECTION = 15

# -------------------- ENTROPIC CONSTRAINTS MATRIX ----------------------------------

RESTRICTIONS = [[[0],[3],[2]],[[0],[3],[2,6]],[[3],[0,6],[2]],[[3],[6],[2]], 
        [[0],[6],[]],
        [[1],[3],[2]],[[1],[3],[2,6]],[[3],[1,6],[2]],
        [[1],[6],[]],
        [[0,1],[3],[2]],[[0,1],[3],[2,6]],[[3],[0,1,6],[2]],
        [[2],[0,1,6]],[[0,1],[6],[]],
        [[2],[4],[3,6]],[[2],[5],[3,6]],[[4],[3,6]],[[5],[3,6]],
        [[0,2],[4],[3,6]],[[0],[4],[3,6]],[[0],[3,4],[2,6]],[[0],[4],[2,6]],
        [[1,2],[4],[3,6]],[[1],[4],[3,6]],[[1],[3,4],[2,6]],[[1],[4],[2,6]],
        [[0,1,2],[4],[3,6]],[[0,1],[4],[3,6]],[[0,1],[3,4],[2,6]],[[0,1],[4],[2,6]],
        [[0,2],[5],[3,6]],[[0],[5],[3,6]],[[0],[3,5],[2,6]],[[0],[5],[2,6]],
        [[1,2],[5],[3,6]],[[1],[5],[3,6]],[[1],[3,5],[2,6]],[[1],[5],[2,6]],
        [[0,1,2],[5],[3,6]],[[0,1],[5],[3,6]],[[0,1],[3,5],[2,6]],[[0,1],[5],[2,6]],
        [[0],[4],[2,3,6]],[[1],[4],[2,3,6]], [[0,1],[4],[2,3,6]],[[0],[5],[2,3,6]],[[1],[5],[2,3,6]], [[0,1],[5],[2,3,6]],
        [[4],[0,2,5],[3,6]],[[4],[1,2,5],[3,6]],[[4],[0,1,2,5],[3,6]],[[5],[0,2,4],[3,6]],[[5],[1,2,4],[3,6]],[[5],[0,1,2,4],[3,6]],
        [[4],[0,5],[3,6]],[[4],[1,5],[3,6]],[[4],[0,1,5],[3,6]],[[5],[0,4],[3,6]],[[5],[1,4],[3,6]],[[5],[0,1,4],[3,6]]]

RESTRICTIONS = RESTRICTIONS + [[[0],[1],[]]]

(basic_inequalities, h_vector_vars) = basic_inequalities(N_VARIABLES)

cir = linear_constraints(RESTRICTIONS, N_VARIABLES)

dimension = len(h_vector_vars)

newrow = np.zeros(dimension, dtype='int')
cir = np.vstack([cir, newrow])
n_cir = len(cir)

cir[n_cir - 1][h_vector_vars.index((0,1,6))] = 1
cir[n_cir - 1][h_vector_vars.index((0,))] = -1
cir[n_cir - 1][h_vector_vars.index((1,))] = -1
cir[n_cir - 1][h_vector_vars.index((6,))] = -1

inequalities = np.vstack((cir, -cir, basic_inequalities))

# ----------------- SELECTING INEQUALITIES ----------------------------------------------

marginal_h_vars = marginal_scenario(MARGINAL_SCENARIO_SELECTION, N_VARIABLES)[0]

info_indices = [h_vector_vars.index(i) for i in MUTUAL_INFORMATION_VARIABLES]

newrow = np.zeros(dimension, dtype='int')
constraints = np.array([newrow])

j = 0
for k in marginal_h_vars:
    constraints[j][h_vector_vars.index(k)] = 1
    constraints = np.vstack([constraints, newrow])
    j = j + 1
constraints = np.delete(constraints, (len(constraints)-1), axis = 0)

with open(CLASS_MATRIX_FILE, "rb") as f:
    class_matrix = pickle.load(f)
    
for N_CLASS in range(START_CLASS, NUMBER_OF_CLASSES):
        
    bound, opt_vertex, opt_h = optimal_vertex(info_indices, inequalities, constraints, marginal_h_vars, MUTUAL_INFORMATION_VARIABLES, class_matrix, N_CLASS)

    round_count = 0
    no_stop = True
    multiplier = 0
    current_bound = 2.0
    atol = 1e-5
    while no_stop:
        
        round_count += 1
        
        start_time = time.perf_counter()

        selected_inequalities = eliminate(info_indices, inequalities, constraints, bound, opt_h)
        log_selected_inequalities = len(selected_inequalities)

        # ------------------ FOURIER-MOTZKIN ELIMINATION ---------------------------------------------

        n_selected_inequalities = len(selected_inequalities)
        zeros_vector = np.zeros((n_selected_inequalities, 1))

        (marginal_h_vars, elim_indices) = marginal_scenario(MARGINAL_SCENARIO_LAMBDA, N_VARIABLES)

        n_vars_to_eliminate = len(elim_indices) 

        count = 0
        for var_index in elim_indices:
                
            print('Current elimination:', count + 1, 'Total: ', n_vars_to_eliminate)

            (new_inequalities, new_zeros, data) = fourier_motzkin_eliminate_single(var_index, selected_inequalities, zeros_vector)
            
            selected_inequalities = new_inequalities
            zeros_vector = new_zeros
            
            count += 1
            
        n_selected_inequalities = len(selected_inequalities)
        dimension = len(selected_inequalities[0])
        zeros_vector = np.zeros((n_selected_inequalities, 1))

        print(f"Result: {n_selected_inequalities} inequalities ----- dimension {dimension}")

        (marginal_h_vars, elim_indices) = marginal_scenario_2(marginal_h_vars, MARGINAL_SCENARIO_FINAL, N_VARIABLES)

        n_vars_to_eliminate = len(elim_indices)

        count = 0
        for var_index in elim_indices:
                
            print('Current elimination:', count + 1, 'Total: ', n_vars_to_eliminate)

            (new_inequalities, new_zeros, data) = fourier_motzkin_eliminate_single(var_index, selected_inequalities, zeros_vector)
            
            selected_inequalities = new_inequalities
            zeros_vector = new_zeros
            
            count += 1
            
        n_selected_inequalities = len(selected_inequalities)
        dimension = len(selected_inequalities[0])
        zeros_vector = np.zeros((n_selected_inequalities, 1))

        print(f"Result: {n_selected_inequalities} inequalities ----- dimension {dimension}")

        # ------------------ CHECKING REDUNDANCY WITH BASICS -------------------------------------------------

        total_dimension = len(basic_inequalities[0])

        final_inequalities = np.zeros((n_selected_inequalities, total_dimension), dtype='float')

        for i in range(0, n_selected_inequalities):
            for j in range(0, dimension):
                if(selected_inequalities[i][j] != 0.0):
                    final_inequalities[i][h_vector_vars.index(marginal_h_vars[j])] = selected_inequalities[i][j]
        
        final_inequalities = compare(final_inequalities, basic_inequalities)
        n_final_inequalities = len(final_inequalities)

        inequality_matrix = np.zeros((n_final_inequalities, dimension), dtype='float')

        for i in range(0, n_final_inequalities):
            for j in range(0, total_dimension):
                if(final_inequalities[i][j] != 0.0):
                    inequality_matrix[i][marginal_h_vars.index(h_vector_vars[j])] = final_inequalities[i][j]

        n_inequalities = len(inequality_matrix)
        print(f"Non-trivial: {n_final_inequalities}")
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # ------------------ TRANSLATION AND OUTPUT -------------------------------------------------
        
        with open(f"results/class_{N_CLASS}/all_inequalities_class_{N_CLASS}.txt", "a") as file:
            np.savetxt(file, inequality_matrix, fmt = "%s")        
            
        all_inequalities = np.loadtxt(f"results/class_{N_CLASS}/all_inequalities_class_{N_CLASS}.txt", ndmin = 2)
        unique_inequalities, counts = np.unique(all_inequalities, axis = 0, return_counts = True)
        
        n_unique_inequalities = len(unique_inequalities)
        
        with open(f"results/class_{N_CLASS}/counts_class_{N_CLASS}.txt", "w") as file:
            for i, count in enumerate(counts, start=1):
                file.write(f"Inequality {i}: {count} repetitions\n")
            
        with open(f"results/class_{N_CLASS}/unique_inequalities_class_{N_CLASS}.txt", "w") as file:
            np.savetxt(file, unique_inequalities, fmt = "%s")
            
        log_translated_list = traduz(inequality_matrix, NAMES, elim_indices)
        
        lines = []
        for row in log_translated_list:
            row = [f"{coef}{''.join(vars)}" for coef, vars in row]
            line = '+'.join(row).replace('+-', '-')
            lines.append(line)
            
        log_translated = '\n'.join(lines)
        
        log_translated = re.sub(r'(?<![\d.])1\.0', '', log_translated)
        log_translated = re.sub(r'(?<![\d.])-1\.0', '-', log_translated)

        translated_list = traduz(unique_inequalities, NAMES, elim_indices)

        lines = []
        for row in translated_list:
            row = [f"{coef}{''.join(vars)}" for coef, vars in row]
            line = '+'.join(row).replace('+-', '-')
            lines.append(line)

        translated = '\n'.join(lines)

        translated = re.sub(r'(?<![\d.])1\.0', '', translated)
        translated = re.sub(r'(?<![\d.])-1\.0', '-', translated)

        with open(f"results/class_{N_CLASS}/translated_inequalities_class_{N_CLASS}.txt", "w") as file:
            file.write(translated)
            
        with open(f"results/class_{N_CLASS}/log_selection_class_{N_CLASS}.txt", "a") as file:
            file.write(f"\n-------- ORDERING {round_count} --------\n")
            file.write(f"\nSelected ineqaulities: {log_selected_inequalities}\n")
            file.write(f"\nNon-trivial: {n_inequalities}\n")
            file.write(log_translated)
            file.write(f"\nTotal unique inequalities: {n_unique_inequalities}\n")
            file.write(f"Time taken: {elapsed_time}\n")
            
        # --------------- STOPPING CRITERIA -----------------------------
                
        if round_count == (BASELINE + multiplier*INCREASE):
            
            gamma_bound, best_inequalities, best_vertex = minimize_class(marginal_h_vars, MUTUAL_INFORMATION_VARIABLES, unique_inequalities, N_CLASS, N_EPS, N_BISECTION, class_matrix)
            
            opt_inequalities = unique_inequalities[best_inequalities]
            
            opt_translated_list = traduz(opt_inequalities, NAMES, elim_indices)
                
            lines = []
            for row in opt_translated_list:
                row = [f"{coef}{''.join(vars)}" for coef, vars in row]
                line = '+'.join(row).replace('+-', '-')
                lines.append(line)
                
            opt_translated = '\n'.join(lines)
            
            opt_translated = re.sub(r'(?<![\d.])1\.0', '', opt_translated)
            opt_translated = re.sub(r'(?<![\d.])-1\.0', '-', opt_translated)
            
            with open(f"results/class_{N_CLASS}/log_bounds_class_{N_CLASS}.txt", "a") as file:
                    file.write(f"\n-------- CALCULATE BOUNDS {multiplier + 1} --------\n")
                    file.write(f"\nBound: {gamma_bound}\n")
                    file.write(f"\nUsed inequalities: {n_unique_inequalities}\n")                
                    file.write(f"Best inequalities: {best_inequalities}\n")
                    file.write(opt_translated)
                    file.write(f"\nOptimal vertex: {best_vertex}\n")
                    
            if gamma_bound < current_bound - atol:
                current_bound = gamma_bound
                multiplier += 1
            
            else:
                no_stop = False