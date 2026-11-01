import random
import numpy as np
import gurobipy as gp

from auxiliary_functions import H_vector_4422_class

def calculate_bound(info_indices, inequalities, constraints, marginal_h_vector):

    n_inequalities = len(inequalities)
    dimension = len(inequalities[0])

    zero_column = np.zeros(n_inequalities)

    info_constraints = np.zeros(dimension)
    info_constraints[info_indices[0]] = 1
    info_constraints[info_indices[1]] = 1
    info_constraints[info_indices[2]] = -1
     
    problem = gp.Model()
    
    variables = problem.addMVar(dimension, lb = 0.0, ub = gp.GRB.INFINITY)

    problem.setObjective(info_constraints @ variables, gp.GRB.MINIMIZE)
    problem.addConstr(inequalities @ variables <= zero_column)    
    problem.addConstr((constraints @ variables) == marginal_h_vector)

    problem.update()
    problem.setParam("OutputFlag", False)
    problem.optimize()
    
    if problem.Status == gp.GRB.OPTIMAL:
        result = problem.ObjVal
        return result
        
    else:
        print("Unbounded!")
        return 0
    
def optimal_vertex(info_indices, inequalities, constraints, marginal_h_vars, mutual_info_vars, class_matrix, n_class):
    
    class_size = len(class_matrix[n_class])
    current_bound = -1.0
    current_vertex = 0
    for n_vertex in range(class_size):
        marginal_h_vector = H_vector_4422_class(marginal_h_vars, mutual_info_vars, n_class, n_vertex, 0.0, 1.0, class_matrix)[0]
        bound = calculate_bound(info_indices, inequalities, constraints, marginal_h_vector)
        
        print(f"Testing vertex {n_vertex} out of {class_size} ----- bound: {bound}")
        
        atol = 1e-8
        if bound > current_bound + atol:
            current_bound = bound
            current_vertex = n_vertex
            current_h_vector = marginal_h_vector
                                    
        if current_bound == 2.0:
            print(f"Best bound: {current_bound}")
            return current_bound, current_vertex, current_h_vector
        
    print(f"Best bound: {current_bound}")
        
    return current_bound, current_vertex, current_h_vector
    
def eliminate(info_indices, inequalities, constraints, bound, opt_h):

    n_inequalities = len(inequalities)
    dimension = len(inequalities[0])

    zero_column = np.zeros(n_inequalities)

    info_constraints = np.zeros(dimension)
    info_constraints[info_indices[0]] = 1
    info_constraints[info_indices[1]] = 1
    info_constraints[info_indices[2]] = -1

    # Test every original row once, in random order.
    rows_to_test = list(range(n_inequalities))
    random.shuffle(rows_to_test)

    # Track the original indices of the rows still present.
    active_rows = list(range(n_inequalities))

    round = 1
    while round <= n_inequalities:
        
        print(f"Testing inequality {round} out of {n_inequalities}")

        original_row = rows_to_test[round - 1]
        random_row = active_rows.index(original_row)

        reduced_ineq = np.delete(inequalities, (random_row), axis = 0)
        reduced_zeros = np.delete(zero_column, (random_row), axis = 0)

        problem = gp.Model()
        problem.setParam("OutputFlag", False)
        problem.setParam("DualReductions", 0)
        
        variables = problem.addMVar(dimension, lb=0, ub=gp.GRB.INFINITY)
        problem.setObjective(info_constraints @ variables, gp.GRB.MINIMIZE)
        problem.addConstr(reduced_ineq @ variables <= reduced_zeros)
        problem.addConstr(constraints @ variables == opt_h)

        problem.update()  
        problem.optimize()

        atol = 10**-5
        if problem.Status == gp.GRB.OPTIMAL:

            if not (problem.ObjVal < bound - atol):
                inequalities = reduced_ineq
                zero_column = reduced_zeros
                del active_rows[random_row]
                
        elif problem.Status == gp.GRB.UNBOUNDED:
            pass

        else:
            print(f"LP did not solve to optimality; status={problem.Status}")

        round += 1
        
    n_selected = len(inequalities)
    print(f"Result: {n_selected} selected inequalities")

    return inequalities
