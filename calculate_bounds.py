import numpy as np
from auxiliary_functions import H_vector_4422_class
        
def test_violation_class(marginal_h_vars, info_vars, inequalities, n_class, n_vertex, eps, gamma, class_matrix):

    tol = 1e-14

    n_inequalities = len(inequalities)

    (H, bound) = H_vector_4422_class(marginal_h_vars, info_vars, n_class, n_vertex, eps, gamma, class_matrix)
    H_vector = np.transpose(np.array(H))

    value_inequalities = inequalities @ H_vector

    violated_inequalities = []

    for i in range(n_inequalities):
        if value_inequalities[i] > tol:
            violated_inequalities.append(i)

    return violated_inequalities


def iterate_eps(marginal_h_vars, info_vars, inequalities, n_class, n_vertex, n_eps, gamma, class_matrix):

    # eps_values = np.linspace(0.0, 0.50001, num=n_eps)
    eps_values = [0.50001]

    all_violated = []
    for eps in eps_values:
        test_violation = test_violation_class(marginal_h_vars, info_vars, inequalities, n_class, n_vertex, eps, gamma, class_matrix)
        all_violated += test_violation

    return list(dict.fromkeys(all_violated))


def gamma_bound(marginal_h_vars, info_vars, inequalities, n_class, n_vertex, n_eps, n_gamma, class_matrix):

    gamma_value = 0.5
    gamma_step = gamma_value / 2
    last_violated = []

    for _ in range(n_gamma):

        violated = iterate_eps(marginal_h_vars, info_vars, inequalities, n_class, n_vertex, n_eps, gamma_value, class_matrix)

        if len(violated) == 0:
            gamma_value += gamma_step

        else:
            gamma_value -= gamma_step
            last_violated = violated

        gamma_step = gamma_step / 2

    return gamma_value, last_violated

def minimize_class(marginal_h_vars, info_vars, inequalities, n_class, n_eps, n_gamma, class_matrix):

    class_size = len(class_matrix[n_class])

    minimal_value = 1.0
    violated = []
        
    for vertex in range(class_size):
        
        print("class: ",n_class,"--- testing ",vertex,"out of",class_size,)
        
        value, last_violated = gamma_bound(marginal_h_vars, info_vars, inequalities, n_class, vertex, n_eps, n_gamma, class_matrix)
                
        if value < minimal_value:
            minimal_value = value
            violated = last_violated
            optimal_vertex = vertex
                        
    return minimal_value, violated, optimal_vertex