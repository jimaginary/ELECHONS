import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EARTH_RADIUS = 6371.0

def earth_distance(lat1, long1, lat2, long2):
    lat1, long1, lat2, long2 = map(np.radians, [lat1, long1, lat2, long2])
    
    dlat = lat1 - lat2
    dlong = long1 - long2

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = EARTH_RADIUS * c

    return distance

def distance_matrix(stations):
    n = len(stations)
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i,j] = earth_distance(stations.iloc[i]['lat'], stations.iloc[i]['long'], stations.iloc[j]['lat'], stations.iloc[j]['long'])
    return M

# M a distance matrix and K the number of nearest neighbours
def K_nearest(M, K, undirected=True):
    B = np.zeros_like(M)
    for i in range(M.shape[0]):
        # select K+1 smallest to bypass vertex and itself being a smallest entry
        B[i, np.argpartition(M[i], K + 1)[:K+1]] = np.ones(K+1)
    if undirected:
        # force matrix symmetry (not perfect K nearest)
        B = np.minimum(1, B + B.T)
    # remove vertex-self connections
    np.fill_diagonal(B, 0)
    return B

def filtered_distance_matrix(stations, K):
    M = distance_matrix(stations)
    return M*K_nearest(M, K)

def closeness(v, scale):
    return np.sqrt(np.sum(np.exp(-np.pow(v[v != 0]/scale,2))))

def closeness_matrix(stations, scale, K):
    n = len(stations)
    M = filtered_distance_matrix(stations, K)
    vertex_weights = np.array([closeness(np.array(v), scale) for v in M.tolist()])

    W = np.zeros_like(M)
    for i in range(n):
        for j in range(n):
            if M[i,j] != 0:
                W[i,j] = closeness(M[i,j], scale)**2/(vertex_weights[i]*vertex_weights[j])
    return W

def laplacian_matrix(stations, K):
    n = len(stations)
    A = filtered_distance_matrix(stations, K)
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if A[i,j] != 0:
                M[i,j] = 1 / A[i,j]
            
    D = np.diag(np.sum(M, 0))
    return D - M
