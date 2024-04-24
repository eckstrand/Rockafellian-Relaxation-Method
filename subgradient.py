'''
Implementation by Gabriel Custodio Rangel
gabriel.custodiorangel.br@nps.edu
'''

import numpy as np
import pyomo.environ as pyo

def subgradient(sub_iterations,nobs_trn,theta,p,lam,u,l):
    
    q = np.array([p_i + u_i for p_i, u_i in zip(p, u)])
    
    J = list(range(nobs_trn))
    I = list(reversed(range(nobs_trn)))
    # creating the vector q,a:
    
    a = np.array([0.0]*nobs_trn)
    pen = np.array([0.0]*nobs_trn)

    # u = np.array(u)  # Ensure u is a numpy array
    # u_previous = np.copy(u)  # store initial u values
    # num_negatives_previous = np.sum(u < 0)  # number of negative values in the initial u



    for n in range(sub_iterations):

        #Step 1
        count = 0

        while count < sub_iterations :

            for j in J:
                if q[j] > p[j]:
                    pen[j] = theta
                elif q[j] == p[j]:
                    pen[j] = 0.0
                elif q[j] < p[j]:
                    pen[j] = -theta
                a[j] = l[j] + pen[j]

            #Step 2
            q = q - lam * a
            #Step 3
            q_hs = np.sort(q)[::-1]


            it=0
            for i in I:
                if (1 - (sum(q_hs[j] for j in range(len(J)-it))/(len(I)-it) + q_hs[i])) > 0:
                    delta = (1 - (sum(q_hs[j] for j in range(len(J)-it))))/(len(I) - it)
                    break
                else:
                    it += 1

            for j in J:
                q[j]= max(q[j]+delta,0)

            #Step 4
            count +=1
            u = q - p

        # num_negatives_current = np.sum(u < 0)  # number of negative values in current u

        # if num_negatives_current < num_negatives_previous:
        #     u = np.copy(u_previous)  # Use the u values from the previous iteration
        # else:
        #     u_previous = np.copy(u)  # Update u_previous for the next iteration
        #     num_negatives_previous = num_negatives_current  # Update num_negatives for the next iteration
     
        # if not np.all(u == 0):
        # # Scale u values and ensure the sum is zero
        #     scale_factor = (1/nobs_trn) / max(abs(u.max()), abs(u.min()))
        #     u = u * scale_factor
        #     u -= u.mean()

        # Try to scale #2

        # min_val = np.min(u)
        # max_val = np.max(u)
        # range_val = max_val - min_val

        # if range_val != 0:  # Avoid dividing by zero
        #     u = (u - min_val) / range_val  # normalize to [0,1]
        #     u = (u - 0.5) * (2/nobs_trn)  # scale to [-1/nobs_trn, 1/nobs_trn]
    
        # mean_val = np.mean(u)
        # u -= mean_val  # adjust to make sum equal to 0

        # Start of new rescaling process:
        min_val = np.min(u)
        max_val = np.max(u)
        sum_neg = np.sum(u[u < 0])

        # Set the most negative value to -1/nobs_trn
        u[u == min_val] = -1/nobs_trn

        # Set the most positive value to the sum of all negative values
        u[u == max_val] = sum_neg

        # Set all other positive values to zero
        u[(u > 0) & (u != sum_neg)] = 0

        # Calculate the current sum of u
        current_sum = np.sum(u)

        # Determine the values in the middle (which are now negative)
        mid_values = u[(u != 0) & (u != -1/nobs_trn)]

        # Distribute the negative (or positive) of the current sum evenly among the middle values
        if len(mid_values) > 0:  # Avoid dividing by zero
            mid_values -= current_sum / len(mid_values)

        # Assign the adjusted middle values back to u
        u[(u != 0) & (u != -1/nobs_trn)] = mid_values



        # Centering q around zero
        # Scaling non-zero u values by their absolute sum
        # non_zero_indices = np.nonzero(u)[0]
        # avg_non_zero = np.mean(u[non_zero_indices])
        # u[non_zero_indices] -= avg_non_zero
        # # Define scale factor correlated to nobs_trn
        # scale_factor = 1.0 / nobs_trn

        # # Apply scale factor to non-zero values
        # u[non_zero_indices] = u[non_zero_indices] * scale_factor
    
    return u.tolist()

def lpbased(u,mu,u_prev):
    u =  [mu*u_new + (1-mu)*u_prv for u_new, u_prv in zip(u, u_prev)]
    return u

def pyo_model(p, losses, alpha, beta, theta, eta, u_reg, label):
    '''
    Assumes `label` is an array of arrays. Each element of the array is the 1-hot encoding of the class. For example,
    if there are 3 classes, then the class "1" would be encoded as [0, 1, 0] and the class "0" would be encoded as
    [1, 0, 0].
    '''
    N = len(p)
    model = pyo.ConcreteModel()
    model.u = pyo.Var(range(N), initialize=0, domain=pyo.Reals)

    if u_reg == 'l1':
        model.z = pyo.Var(range(N), initialize=0, domain=pyo.NonNegativeReals)

        model.obj = pyo.Objective(
            expr = sum((model.u[i] + p[i])*losses[i] + theta * model.z[i] for i in range(N)),
            sense = pyo.minimize
        )

        def boundL_rule(model, i):
            return -model.u[i] <= model.z[i]
        model.boundL = pyo.Constraint(range(N), rule=boundL_rule)

        def boundU_rule(model, i):
            return model.u[i] <= model.z[i]
        model.boundU = pyo.Constraint(range(N), rule=boundU_rule)

    elif u_reg == 'max':
        model.z = pyo.Var(range(N), initialize=0, domain=pyo.NonNegativeReals)

        model.obj = pyo.Objective(
            expr = sum((model.u[i] + p[i])*losses[i] + model.z[i] for i in range(N)),
            sense = pyo.minimize
        )

        def boundL_rule(model, i):
            if int(label[i][1]) == 1: # class "1"
               return alpha*model.u[i] <= model.z[i]
            else:                     # class "0"
               return eta*model.u[i] <= model.z[i]
        model.boundL = pyo.Constraint(range(N), rule=boundL_rule)

        def boundU_rule(model, i):
            if int(label[i][1]) == 1:
               return beta*model.u[i] <= model.z[i]
            else:
               return theta*model.u[i] <= model.z[i]
        model.boundU = pyo.Constraint(range(N), rule=boundU_rule)


    elif u_reg == 'l2':
        model.obj = pyo.Objective(
            expr = sum((model.u[i] + p[i])*losses[i] + 0.5 * theta * model.u[i]**2 for i in range(N)),
            sense = pyo.minimize
        )
    elif u_reg == 'lp':
        model.v = pyo.Var(range(N), initialize=0, domain=pyo.Reals)

        model.obj = pyo.Objective(
        expr = sum(model.u[i] *losses[i] + theta * model.v[i] for i in range(N)),
        sense = pyo.minimize
    )

        def boundL_rule(model, i):
            return -model.u[i] <= model.v[i]
        model.boundL = pyo.Constraint(range(N), rule=boundL_rule)

        def boundU_rule(model, i):
            return model.u[i] <= model.v[i]
        model.boundU = pyo.Constraint(range(N), rule=boundU_rule)



    else:
        raise Exception("Unsupported u-regularization = {}".format(u_reg))

    def rule1(model, i):
        return model.u[i] + p[i] >= 0
    model.c1 = pyo.Constraint(range(N), rule=rule1)

    model.c2 = pyo.Constraint(expr = sum(model.u[i] for i in range(N)) == 0)




    return model
