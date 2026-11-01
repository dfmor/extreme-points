import math
import time
import numpy as np
import itertools as itt
import gurobipy as gp

def basic_inequalities(n, M=None):
    """Build the basic inequality matrix and its associated entropy vector."""
    if M == None:
        variable_indices = np.arange(n)

        # Generate all nonempty combinations of variable indices.
        combinations = [
            list(itt.combinations(variable_indices, size))
            for size in range(1, n + 1)
        ]
        entropy_vector = list(itt.chain(*combinations))
    else:
        entropy_vector = M

    inequality_matrix = np.zeros((1, len(entropy_vector)), dtype="int")
    empty_row = np.zeros(len(entropy_vector), dtype="int")
    row_index = 0

    # Generate conditional entropy inequalities:
    # -H(A | B) = -H(A, B) + H(B).
    for first_variable in variable_indices:
        joint_term = tuple(np.sort(variable_indices))

        conditioning_variables = np.delete(
            variable_indices,
            np.where(variable_indices == first_variable)[0][0],
        )
        conditioning_term = tuple(np.sort(conditioning_variables))

        column_index = entropy_vector.index(joint_term)
        inequality_matrix[row_index][column_index] = -1

        column_index = entropy_vector.index(conditioning_term)
        inequality_matrix[row_index][column_index] = 1

        # Append an empty row for the next inequality.
        inequality_matrix = np.vstack([inequality_matrix, empty_row])
        row_index = row_index + 1

    # Generate conditional and unconditional mutual information inequalities.
    for first_variable in variable_indices:
        remaining_variables = np.delete(
            variable_indices,
            np.where(variable_indices <= first_variable)[0],
        )

        for second_variable in remaining_variables:
            conditioning_variables = np.delete(
                variable_indices,
                np.where(variable_indices == first_variable)[0][0],
            )
            conditioning_variables = np.delete(
                conditioning_variables,
                np.where(conditioning_variables == second_variable)[0][0],
            )

            # Generate conditional mutual information inequalities:
            # -I(A : B | C) = -H(A, C) - H(B, C) + H(A, B, C) + H(C).
            if np.size(conditioning_variables) != 0:
                conditioning_combinations = [
                    list(itt.combinations(conditioning_variables, size))
                    for size in range(1, len(conditioning_variables) + 1)
                ]
                conditioning_combinations = list(
                    itt.chain(*conditioning_combinations)
                )

                for conditioning_term in conditioning_combinations:
                    first_conditioned_term = np.concatenate(
                        ((first_variable,), conditioning_term)
                    ).astype(int)
                    first_conditioned_term = tuple(
                        np.sort(first_conditioned_term)
                    )

                    column_index = entropy_vector.index(
                        first_conditioned_term
                    )
                    inequality_matrix[row_index][column_index] = -1

                    second_conditioned_term = np.concatenate(
                        ((second_variable,), conditioning_term)
                    ).astype(int)
                    second_conditioned_term = tuple(
                        np.sort(second_conditioned_term)
                    )

                    column_index = entropy_vector.index(
                        second_conditioned_term
                    )
                    inequality_matrix[row_index][column_index] = -1

                    joint_term = np.concatenate(
                        (
                            (first_variable,),
                            (second_variable,),
                            conditioning_term,
                        )
                    ).astype(int)
                    joint_term = tuple(np.sort(joint_term))

                    column_index = entropy_vector.index(joint_term)
                    inequality_matrix[row_index][column_index] = 1

                    column_index = entropy_vector.index(conditioning_term)
                    inequality_matrix[row_index][column_index] = 1

                    inequality_matrix = np.vstack(
                        [inequality_matrix, empty_row]
                    )
                    row_index = row_index + 1

            # Generate unconditional mutual information inequalities:
            # -I(A : B) = -H(A) - H(B) + H(A, B).
            column_index = entropy_vector.index(first_variable)
            inequality_matrix[row_index][column_index] = -1

            column_index = entropy_vector.index(second_variable)
            inequality_matrix[row_index][column_index] = -1

            joint_term = np.concatenate(
                ((first_variable,), (second_variable,))
            ).astype(int)
            joint_term = tuple(np.sort(joint_term))

            column_index = entropy_vector.index(joint_term)
            inequality_matrix[row_index][column_index] = 1

            inequality_matrix = np.vstack([inequality_matrix, empty_row])
            row_index = row_index + 1

    # Remove the final empty row and duplicate inequalities.
    inequality_matrix = np.delete(
        inequality_matrix,
        len(inequality_matrix) - 1,
        axis=0,
    )
    inequality_matrix = np.unique(inequality_matrix, axis=0)

    return inequality_matrix, entropy_vector

def marginal_scenario(CM, n):
    """Build the marginal entropy vector and identify excluded indices."""
    variable_indices = np.arange(n)

    # Generate all nonempty combinations of variable indices.
    combinations = [
        list(itt.combinations(variable_indices, size))
        for size in range(1, n + 1)
    ]
    entropy_vector = list(itt.chain(*combinations))

    marginal_terms = []
    excluded_indices = np.array([len(entropy_vector) + 1])

    # Collect all nonempty subsets of each marginal context.
    for context in CM:
        combinations = [
            list(itt.combinations(context, size))
            for size in range(1, len(context) + 1)
        ]
        context_terms = list(itt.chain(*combinations))
        marginal_terms = list(set(itt.chain(marginal_terms, context_terms)))

    # Find entropy-vector indices that are absent from the marginal terms.
    for term in entropy_vector:
        if marginal_terms.count(term) == 0:
            if excluded_indices[0] == len(entropy_vector) + 1:
                excluded_indices[0] = entropy_vector.index(term)
            else:
                excluded_indices = np.concatenate(
                    (
                        excluded_indices,
                        np.array([entropy_vector.index(term)]),
                    )
                )

    # Sort indices in descending order so deletions preserve remaining indices.
    sorted_excluded_indices = np.sort(excluded_indices)[::-1].astype(int)

    # Preserve the original entropy-vector order in the marginal scenario.
    marginal_terms = entropy_vector
    for index in sorted_excluded_indices:
        del marginal_terms[index]

    return marginal_terms, sorted_excluded_indices

def marginal_scenario_2(vetor_M, CM, n):
    """Filter the supplied entropy vector in place by marginal contexts."""
    marginal_terms = []
    excluded_indices = np.array([len(vetor_M) + 1])

    # Collect all nonempty subsets of each marginal context.
    for context in CM:
        combinations = [
            list(itt.combinations(context, size))
            for size in range(1, len(context) + 1)
        ]
        context_terms = list(itt.chain(*combinations))
        marginal_terms = list(set(itt.chain(marginal_terms, context_terms)))

    # Find indices of terms that are absent from the marginal contexts.
    for term in vetor_M:
        if marginal_terms.count(term) == 0:
            if excluded_indices[0] == len(vetor_M) + 1:
                excluded_indices[0] = vetor_M.index(term)
            else:
                excluded_indices = np.concatenate(
                    (
                        excluded_indices,
                        np.array([vetor_M.index(term)]),
                    )
                )

    # Delete in descending index order to preserve the remaining indices.
    sorted_excluded_indices = np.sort(excluded_indices)[::-1].astype(int)

    marginal_terms = vetor_M
    for index in sorted_excluded_indices:
        del marginal_terms[index]

    return marginal_terms, sorted_excluded_indices

def linear_constraints(I, n):
    """Build a matrix of linear constraints from entropy expressions."""
    variable_indices = np.arange(n)

    # Generate all nonempty combinations of variable indices.
    combinations = [
        list(itt.combinations(variable_indices, size))
        for size in range(1, n + 1)
    ]
    entropy_vector = list(itt.chain(*combinations))

    constraint_matrix = np.zeros((1, len(entropy_vector)), dtype="int")
    empty_row = np.zeros(len(entropy_vector), dtype="int")
    row_index = 0

    for constraint in I:
        # Mutual information: I(A : B | C).
        if len(constraint) == 3:
            # H(A, C)
            entropy_term = np.concatenate(
                (constraint[0], constraint[2])
            ).astype(int)
            entropy_term = tuple(np.sort(entropy_term))
            column_index = entropy_vector.index(entropy_term)
            constraint_matrix[row_index][column_index] = 1

            # H(B, C)
            entropy_term = np.concatenate(
                (constraint[1], constraint[2])
            ).astype(int)
            entropy_term = tuple(np.sort(entropy_term))
            column_index = entropy_vector.index(entropy_term)
            constraint_matrix[row_index][column_index] = 1

            # -H(A, B, C)
            entropy_term = np.concatenate(
                (constraint[0], constraint[1], constraint[2])
            ).astype(int)
            entropy_term = tuple(np.sort(entropy_term))
            column_index = entropy_vector.index(entropy_term)
            constraint_matrix[row_index][column_index] = -1

            # Include -H(C) only when the conditioning term is nonempty.
            if len(constraint[2]) > 0:
                entropy_term = tuple(np.sort(constraint[2]))
                column_index = entropy_vector.index(entropy_term)
                constraint_matrix[row_index][column_index] = -1

            # Append an empty row unless this is the final constraint.
            if row_index + 1 != len(I):
                constraint_matrix = np.vstack([constraint_matrix, empty_row])
            row_index = row_index + 1

        # Conditional entropy: H(A | B).
        if len(constraint) == 2:
            # H(A, B)
            entropy_term = np.concatenate(
                (constraint[0], constraint[1])
            ).astype(int)
            entropy_term = tuple(np.sort(entropy_term))
            column_index = entropy_vector.index(entropy_term)
            constraint_matrix[row_index][column_index] = 1

            # -H(B)
            entropy_term = tuple(np.sort(constraint[1]))
            column_index = entropy_vector.index(entropy_term)
            constraint_matrix[row_index][column_index] = -1

            # Append an empty row unless this is the final constraint.
            if row_index + 1 != len(I):
                constraint_matrix = np.vstack([constraint_matrix, empty_row])
            row_index = row_index + 1

    return constraint_matrix

def marginal(P, marg):
	n =P.ndim
	x=np.arange(n)
	
	elim = np.delete(x, marg)
	elim = np.sort(elim)[::-1].astype(int)
	
	for i in elim:
		P = P.sum(axis=i,dtype='float')
	
	return P

def Index4422(variables):
    a = variables[0]
    b = variables[1]
    x = variables[2]
    y = variables[3]
    
    index = 32*a + 16*b + 4*x + y
    return index  

def probability_class(a, b, x, y, n_class, n_vertex, class_matrix):    
    index = Index4422((a, b, x, y))
    point = class_matrix[n_class][n_vertex]
    return point[index]

def behavior_class(n_class, n_vertex, gamma, class_matrix):

    p = np.zeros((2, 2, 4, 4), dtype="float")  # p(a,b|x,y)

    for a in [0, 1]:
        for b in [0, 1]:
            for x in [0, 1, 2, 3]:
                for y in [0, 1, 2, 3]:

                    p[a][b][x][y] = gamma * probability_class(a, b, x, y, n_class, n_vertex, class_matrix) + (1 - gamma) * (1 / 4)

    return p

def code(b0, b1):
    return 2 * b0 + b1

def H(prob):
	S=np.size(prob)
	Pro=np.reshape(prob, S)
	h=np.zeros(S, dtype='float')
	for i in range(0,S):
		if Pro[i]>=10**-16:
			h[i]=-Pro[i]*math.log(Pro[i],2)
	Ent=h.sum()
	return Ent

def H_vector_4422_class(M, opt_func, n_class, n_vertex, eps, gamma, class_matrix):

    Px0x1mabmlG0 = np.zeros(
        (2, 2, 2, 2, 2, 2, 2), dtype="float"
    )  # p(x0,x1,m,a,b,m',g0)
    Px0x1mabmlG1 = np.zeros(
        (2, 2, 2, 2, 2, 2, 2), dtype="float"
    )  # p(x0,x1,m,a,b,m',g1)

    paBDxy = behavior_class(n_class, n_vertex, gamma, class_matrix)  # p(a,b|x,y)

    channelpre = 1
    for pc in np.arange(0, 0.5, 0.5 / channelpre):
        pc = eps 
        pmlDm = np.array([[(1 - pc), pc], [pc, (1 - pc)]])

        for x0, x1, a, ml, B in itt.product(
            range(2), range(2), range(2), range(2), range(2)
        ):  

            Px0x1mabmlG0[x0][x1][a][a][B][ml][B] = (
                (1 / 4) * paBDxy[a][B][code(x0, x1)][code(ml, 0)] * pmlDm[ml][a]
            )
            Px0x1mabmlG1[x0][x1][a][a][B][ml][B] = (
                (1 / 4) * paBDxy[a][B][code(x0, x1)][code(ml, 1)] * pmlDm[ml][a]
            )

        Px0x1mmlG0 = marginal(Px0x1mabmlG0, (0, 1, 2, 5, 6))  # p(x0,x1,m,m',g0)
        Px0x1mmlG1 = marginal(Px0x1mabmlG1, (0, 1, 2, 5, 6))  # p(x0,x1,m,m',g1)

        h = np.array(
            [len(M) + 1], dtype="float"
        )  

        for k in M:
            if (
                k.count(5) == 0
            ):  
                p = marginal(Px0x1mmlG0, k)
                if h[0] == len(M) + 1:
                    h[0] = H(p)
                else:
                    h = np.concatenate((h, np.array([H(p)])))
            else:
                j = list(k)
                j[j.index(5)] = 4
                j = tuple(j)
                p = marginal(Px0x1mmlG1, j)
                if h[0] == len(M) + 1:
                    h[0] = H(p)
                else:
                    h = np.concatenate((h, np.array([H(p)])))

        h_aa = (H(marginal(Px0x1mmlG0, opt_func[0])) + H(marginal(Px0x1mmlG0, opt_func[1])) - H(marginal(Px0x1mmlG0, opt_func[2])))

    return h, h_aa

def test_redundancy(ineq, b, A, B):
    
    if(len(A)!=0):
        B = np.reshape(B, (len(B)))
        b = np.reshape(b, (len(b)))
    
        prob = gp.Model()
        h = prob.addMVar(len(A[0]), lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY )
        prob.setObjective(ineq @ h, gp.GRB.MAXIMIZE)
        prob.addConstr(A @ h <= B)
        prob.update()
        prob.setParam('OutputFlag',False)
        prob.optimize()
    
        atol=10**-8
        B = np.reshape(B, (np.size(B),1))
        b = np.reshape(b, (np.size(b),1))
        if((prob.Status == 2)):
            if(prob.ObjVal>=atol):
                A=np.vstack((A,ineq))
                B=np.vstack((B,b))
        else:
            A=np.vstack((A,ineq))
            B=np.vstack((B,b))
    else:
        A=np.vstack((A,ineq))
        B=np.vstack((B,b))
        
    return (A, B)

def eliminate_redundancy(A, B):
    if(len(A)!=0):
    
        A = np.unique(A, axis = 0)
        B = np.zeros((len(A)))
  
        cont=0
        c=0
        n=len(A)
        matriz_elemental = np.zeros((len(A[0])))
        l=0
        row=0
        while (row <= len(A)-1):
            if(type(n)==type(1)):
                por = int((cont/n)*100)
                cont=cont+1
            a_row = A[row][:]
            b_row = B[row]
            A_row = np.delete(A, (row), axis = 0)
            B_row = np.delete(B, (row), axis = 0)
    
            prob = gp.Model()
            h = prob.addMVar(len(A[0]), lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY )
            prob.setObjective(a_row @ h, gp.GRB.MAXIMIZE)
            prob.addConstr(A_row @ h <= B_row)
            prob.update()
            prob.setParam('OutputFlag',False)

            prob.optimize()

            atol=10**-8
            if((prob.Status == 2)):
                if(prob.ObjVal<atol):
                    A = A_row
                    B = B_row
                else:
                    row = row + 1
            else:
                row = row + 1  
            c=c+1
        B = np.reshape(B, (np.size(B),1))
        
        H_max=np.amax(abs(A),axis=1)
        A=np.dot(np.diag(1/H_max),A)
        B=np.dot(np.diag(1/H_max),B)
    
    return (A, B)

def fourier_motzkin_eliminate_single(var_index,A,b,C=None,d=None,atol=10**-8):
    """
    Performs Fourier-Motzkin elimination method
    Inputs:
        var_index: an integer between [0,N], the index of variable to be removed, N is the number of variables
        A and b: A x <= b
        C and d (optional): Cx=d
        A,B,c,d should be numpy arrays
    Output:
        A_new and b_new such that A_new x_reduced <=b_new, where x_reduced does not include var_index
    """
    tempo_total = time.time()
    if var_index>A.shape[1]:
        raise("Error: %d is greather the number of variables. Choose a variable index between 0 and %d"%(var_index,A.shape[1]))
    if A.shape[0]!=b.shape[0]:
        raise("Error: number of rows in A: ",A.shape[0]," and b:",b.shape[0]," mismatch")
    if type(C)==type(np.array([1])):
        A=np.vstack((A,C,-C))
        b=np.vstack((b,d,-d))
        return fourier_motzkin_eliminate_single(var_index,A,b,None,None,atol)
    else:
        phi_positive=[i for i in range(A.shape[0]) if A[i,var_index]>=atol] # list of positive var entries
        phi_negative=[i for i in range(A.shape[0]) if A[i,var_index]<=-atol]  # list of negative var entries
        phi_core=[i for i in range(A.shape[0]) if abs(A[i,var_index])<atol]  # list of zero var entries
        
        s_smaller=np.diag(1/A[phi_positive,var_index]) # positive
        s_larger=np.diag(1/A[phi_negative,var_index]) # negative

        A_positive=np.dot(s_smaller,A[phi_positive,:]) # A of postives scaled by var entries
        b_positive=np.dot(s_smaller,b[phi_positive,:])
        A_negative=np.dot(s_larger,A[phi_negative,:])
        b_negative=np.dot(s_larger,b[phi_negative,:])
        
        """ We have A_positive x_other + x_r <= b_positive
        --> We have A_negative x_other + x_r >= b_negative
        --> We have b_postive - b_negative >= (A_neg - A _pos) * x_other (all combinations)
        """

        other=list(range(0,var_index))+list(range(var_index+1,A.shape[1]))
        if phi_core!=[]:
            A_new=A[phi_core,:][:,other]
            b_new=b[phi_core,:]
        else:
            A_new=np.empty((0,A.shape[1]-1))
            b_new=np.empty((0,1))

        for i in range(len(phi_positive)):
            for j in range(len(phi_negative)):
                alpha=(-A_negative[j,other]+A_positive[i,other]).reshape(1,len(other))
                beta=b_positive[i,:]-b_negative[j,:]
                (A_new, b_new) = test_redundancy(alpha, beta, A_new, b_new)
        '''
        if phi_core!=[]:
            A_new=np.vstack((A_new,A[phi_core,:][:,other]))
            b_new=np.vstack((b_new,b[phi_core,:]))
        '''
        data = (len(A_new),(len(phi_positive)*len(phi_negative)))
        (A_new,b_new) = eliminate_redundancy(A_new,b_new)
        return (A_new,b_new,data)

def compare(verif, const): 

    B_verif = np.zeros((len(verif)))
    B_const = np.zeros((len(const)))

    cont=0
    n=len(verif)
    l=0
    row=0
    while (row <= len(verif)-1):
        if(type(n)==type(1)):
            por = int((cont/n)*100)
            cont=cont+1
        a_row = verif[row][:]
        b_row = B_verif[row]
        verif_row = np.delete(verif, (row), axis = 0)
        B_verif_row = np.delete(B_verif, (row), axis = 0)

        prob = gp.Model()
        h = prob.addMVar(len(verif[0]), lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY )
        prob.setObjective(a_row @ h, gp.GRB.MAXIMIZE)
        prob.addConstr(const @ h <= B_const)
        prob.update()
        prob.setParam('OutputFlag',False)
        prob.optimize()
        atol=10**-8
        if((prob.Status == 2)):
            if(prob.ObjVal<atol):
                verif = verif_row
                B_verif = B_verif_row
            else:
                row = row + 1
        else:
            row = row + 1
    B_verif = np.reshape(B_verif, (np.size(B_verif),1))
    '''
    H_max=np.amax(abs(verif),axis=1)
    verif=np.dot(np.diag(1/H_max),verif)
    B_verif=np.dot(np.diag(1/H_max),B_verif)
    '''
    return (verif)

def traduz(A, x, Ind=None, Ind_e=None):

	#####################################################################
	####### GERANDO VETOR ENTROPICO COM LETRAS ##########################

	#x=['Z0','Z1','G0','G1','M','L']
	#x=['Z0','Z1','M', 'L','G0','G1']
	n=len(x)
	comb=[list(itt.combinations(x,i)) for i in range (1,n+1)]
	comb = list(itt.chain(*comb))

	if(Ind.any() != None):
		for i in Ind:
			del comb[i]

	'''
	for i in Ind_e:
		del comb[i]
	'''
	#######################################################################
	####### TRADUZINDO DESIGUALDADES ######################################
	H=[[]]
	h=[]
	
	for a in range(0,len(A)):
		H[a] = [(A[a][i],comb[i]) for i in range(0,len(comb)) if A[a][i]!= 0.0]
		H.append(h)
	#print(tuple(H))

	return H
