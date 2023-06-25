import numpy as np

def mutual_information(x, tau, bins=100):
    n = len(x)
    m1 = m2 = bins
    min_val = min(x)
    max_val = max(x)
    delta1 = (max_val - min_val) / m1
    delta2 = (max_val - min_val) / m2

    # Compute the histogram of x
    hist_x, _ = np.histogram(x[:-tau], bins=m1, range=(min_val, max_val))

    # Compute the joint histogram of x and x with delay tau
    hist_joint, _, _ = np.histogram2d(x[:-tau], x[tau:], bins=[m1, m2], range=[[min_val, max_val], [min_val, max_val]])

    # Calculate probabilities
    p_s = hist_x / (n - tau)
    p_q = hist_x / (n - tau)
    p_s_q = hist_joint / ((n - tau) ** 2)

    # Calculate entropy
    h_s = -np.sum(p_s * np.log2(p_s + np.finfo(float).eps))
    h_q = -np.sum(p_q * np.log2(p_q + np.finfo(float).eps))
    h_s_q = -np.sum(p_s_q * np.log2(p_s_q + np.finfo(float).eps))

    # Calculate mutual information
    mi = h_s + h_q - h_s_q

    return mi

def find_optimal_delay(x, max_delay):
    mi_values = []
    delays = range(1, max_delay + 1)
    for delay in delays:
        mi = mutual_information(x, delay)
        mi_values.append(mi)

    optimal_delay = np.argmin(mi_values) + 1

    return optimal_delay

import numpy as np

def calculate_embedding_dimension(data, tau, max_dimension):
    n = len(data)
    E = np.zeros(max_dimension)
    E_star = np.zeros(max_dimension)
    
    for d in range(1, max_dimension + 1):
        yi = np.array([data[i:i+(d-1)*tau+1:tau] for i in range(n-(d-1)*tau)])
        yn = np.zeros_like(yi)
        a = np.zeros(n-(d-1)*tau)
        
        for i in range(n-(d-1)*tau):
            distances = np.max(np.abs(yi - yi[i]), axis=1)
            distances[i] = np.inf
            nearest_neighbor_index = np.argmin(distances)
            yn[i] = yi[nearest_neighbor_index]
            
            a[i] = np.max(np.abs(yi[i+1] - yn[i+1])) / np.max(np.abs(yi[i] - yn[i]))
        
        E[d-1] = np.mean(a)
        E_star[d-1] = np.mean(np.abs(data[d*tau:] - data[np.arange(n-(d-1)*tau) + yn[:, -1]]))
    
    E1 = E[1:] / E[:-1]
    E2 = E_star[1:] / E_star[:-1]
    
    d0 = np.argmin(np.abs(E1 - 1))
    d0 += 1  # Adjust index to start from 1
    
    return d0, E1, E2

def reconstruct_phase_space(data, delay, embedding_dimension):
    n = len(data)
    m = n - (embedding_dimension - 1) * delay
    phase_space = np.zeros((m, embedding_dimension))

    for i in range(m):
        for j in range(embedding_dimension):
            phase_space[i, j] = data[i + j * delay]

    return phase_space

