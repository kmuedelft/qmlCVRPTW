from gurobipy import *
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Data ----------

# Open the file and read the data
with open('data_small_multiTW.txt', 'r') as f:
    data = f.readlines()

CVRPTW = []
for line in data:
    parts = line.split()
    loc_id = int(parts[0])
    x_coord = int(parts[1])
    y_coord = int(parts[2])
    demand = int(parts[3])
    service_time = int(parts[4])
    num_tw = int(parts[5])
    time_windows = []
    for n in range(num_tw):
        start = int(parts[6 + 2*n])
        end = int(parts[7 + 2*n])
        time_windows.append((start, end))

    CVRPTW.append([loc_id, x_coord, y_coord, demand, service_time] + time_windows)

# Convert list to numpy array for easier manipulation
CVRPTW = np.array(CVRPTW, dtype=object)

Nodes = CVRPTW[:, 0]  # nodes
n = len(Nodes)  # number of nodes

Vehicles = {0, 1} 

xcoord = CVRPTW[:, 1]  # x coordinates
ycoord = CVRPTW[:, 2]  # y coordinates
q = CVRPTW[:, 3]  # demand
service_duration = CVRPTW[:, 4]  # service time

# Extract time windows for each node
time_windows = CVRPTW[:, 5:].tolist()

Q = 130  # vehicle capacity
BIGM = 123456  # big M

# Calculate the distance matrix
distance = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            distance[i, j] = math.sqrt((xcoord[i] - xcoord[j])**2 + (ycoord[i] - ycoord[j])**2)

# ---------- Optimization Model ----------

# Initialize the model
m = Model("CVRPTW_MultiTW")

# ---------- Decision Variables ----------

# Binary variable, 1 if vehicle v travels from i to j, 0 otherwise
edge = {}
for i in Nodes:
    for j in Nodes:
        for v in Vehicles:
            edge[i, j, v] = m.addVar(vtype=GRB.BINARY, lb=0, name='x_%s_%s_%s' % (i, j, v))

# Time at which vehicle v starts service at node i at time window w
service_start = {}
for i in Nodes:
    for v in Vehicles:
            service_start[i, v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='t_%s_%s' % (i, v))

# Binary variable, 1 if node i is visited by vehicle v during time window w
z = {}
for i in Nodes:
    for v in Vehicles:
        for w in range(len(time_windows[int(i)])):
            z[i, v, w] = m.addVar(vtype=GRB.BINARY, lb=0, name='z_%s_%s_%s' % (i, v, w))

# Counter of vehicles
y = {}
for v in Vehicles:
    y[v] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0, name='y_%s' % (y))

m.update()

# ---------- Objective Function ----------

obj = quicksum(distance[i,j]*edge[i,j,v] for i in Nodes for j in Nodes for v in Vehicles)
m.setObjective(obj)
m.ModelSense = GRB.MINIMIZE
m.update()

# ---------- Constraints ----------

# No self loops
con1 = {}
for v in Vehicles:
    for i in Nodes:
        con1[i,i,v] = m.addConstr(
            edge[i,i,v] == 0, 'con1[' + str(i) + ',' + str(i) + ',' + str(v) + ']'
        )

# Link edge and z
con2 = {}
for i in Nodes:
    for v in Vehicles:
        con2[i,j,v] = m.addConstr(
            quicksum(edge[i,j,v] for j in Nodes) == quicksum(z[i, v, w] for w in range(len(time_windows[int(i)]))), 'con2[' + str(i) + ',' + str(j) + ',' + str(v) + ']'
        )

# Each node is visited exactly once
con3 = {}
for i in Nodes:
    if i != 0:
        con3[i,v,w] = m.addConstr(
            quicksum(z[i,v,w] for v in Vehicles for w in range(len(time_windows[int(i)]))) == 1, 'con3[' + str(i) + ',' + str(v) + ',' + str(w) + ']'
        )
    
# each used vehicle starts and ends at the depot
con4 = {}
con5 = {}
for v in Vehicles:
    con4[v] = m.addConstr(
        quicksum(edge[0,i,v] for i in Nodes if i != 0) <= y[v], 'con4[' + str(v) + ']'
    )
    con5[v] = m.addConstr(
        quicksum(edge[i,0,v] for i in Nodes if i != 0) <= y[v], 'con5[' + str(v) + ']'
    )

# Capacity constraint
con6 = {}
for v in Vehicles:
    con6[v] = m.addConstr(
        quicksum(q[i]*z[i,v,w] for i in Nodes for w in range(len(time_windows[int(i)]))) <= Q, 'con6[' + str(v) + ']'
    )

# Flow conservation excluding self-loops
con7 = {}
for i in Nodes:
    for v in Vehicles:
        con7[i, j, v] = m.addConstr(
            quicksum(edge[i, j, v] for j in Nodes) == quicksum(edge[j, i, v] for j in Nodes),
            'con7[' + str(i) + ',' + str(j) + ',' + str(v) + ']'
        )

# time window constraints
con8 = {}
con9 = {}
con10 = {}

for i in Nodes:
    for v in Vehicles:
        for w in range(len(time_windows[int(i)])):
            a_iw, b_iw = time_windows[int(i)][w]
            con8[i,v,w] = m.addConstr(
                service_start[i, v] + BIGM * (1 - z[i, v, w]) >= a_iw, 'con8[' + str(i) + ',' + str(v) + str(w) + ']'
                )
            con9[i,v,w] = m.addConstr(
                service_start[i, v] - BIGM * (1 - z[i, v, w]) <= b_iw, 'con9[' + str(i) + ',' + str(v) + str(w) +']'
                )

for i in Nodes:
    for j in Nodes:
        for v in Vehicles:
            for w in range(len(time_windows[int(i)])):   
                if i != j and i != 0 and j != 0:
                    con10[i,j,v,w] = m.addConstr(
                        service_start[j, v] >= service_start[i, v] + service_duration[int(i)] + distance[i, j] - BIGM * (1 - edge[i, j, v]), 'con10[' + str(i) + ',' + str(j) + ',' + str(v) + ',' + str(w) + ']'
                    )

# ---------- Solve ----------

m.update()
m.write('CVRPTW.lp')
m.Params.TimeLimit = 30600
m.optimize()

