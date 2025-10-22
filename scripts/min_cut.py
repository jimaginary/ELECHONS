import elechons.regress_temp as r
import elechons.models.linear_regression as lr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

r.init('mean')
# s1 = np.random.randint(len(r.long))
# s2 = np.random.randint(len(r.long))
s1 = np.argmin(r.long)
s2 = np.argmax(r.long)
print(s1, s2)

p = 1
L = lr.VAR(r.temps_mean_sin_adj, p)
init = L.param_history.T
cap = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, p, init, alpha = 0.005).CMI()

G = nx.DiGraph()
n = cap.shape[0]

for i in range(n):
    for j in range(n):
        if cap[i, j] > 0:
            G.add_edge(i, j, capacity=cap[i, j])

source = s1
sink = s2
cut_value, partition = nx.minimum_cut(G, source, sink)
reachable, non_reachable = partition

# Split positions into reachable/unreachable
x_reach = [r.long[i] for i in reachable]  # longitude
y_reach = [r.lat[i] for i in reachable]  # latitude
x_non = [r.long[i] for i in non_reachable]
y_non = [r.lat[i] for i in non_reachable]

# Plot nodes
plt.figure(figsize=(6,6))
plt.scatter(x_reach, y_reach, c='red', s=100, label='Reachable')
plt.scatter(x_non, y_non, c='blue', s=100, label='Unreachable')

# Label source and sink
plt.scatter(r.long[source], r.lat[source], c='green', s=150, marker='*', label='Source')
plt.scatter(r.long[sink], r.lat[sink], c='orange', s=150, marker='*', label='Sink')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Reachable / Unreachable nodes from min-cut')
plt.legend()
plt.grid(True)
plt.show()