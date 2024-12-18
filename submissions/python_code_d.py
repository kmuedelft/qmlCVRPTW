
from gurobipy import *
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Data ----------

# Open the file and read the data
with open('data_small_multiTW11.txt', 'r') as f:
    data = []
    for line in f:
        # Skip empty lines
        if not line.strip():
            continue
        
        parts = line.split()
        loc_id = int(parts[0])  # Location ID
        x_coord = int(parts[1])  # X coordinate
        y_coord = int(parts[2])  # Y coordinate
        demand = int(parts[3])  # Demand
        service_time = int(parts[4])  # Service time
        num_tw = int(parts[5])  # Number of time windows
        
        # Extract time windows
        time_windows = []
        for n in range(num_tw):
            start = int(parts[6 + 2 * n])
            end = int(parts[7 + 2 * n])
            time_windows.append((start, end))
        
        # Append the parsed data as a dictionary
        data.append({
            'loc_id': loc_id,
            'x_coord': x_coord,
            'y_coord': y_coord,
            'demand': demand,
            'service_time': service_time,
            'num_time_windows': num_tw,
            'time_windows': time_windows
        })

# Find the maximum number of time windows
max_time_windows = max(len(entry['time_windows']) for entry in data)

# Pad nodes with fewer time windows and update num_time_windows
for entry in data:
    while len(entry['time_windows']) < max_time_windows:
        entry['time_windows'].append((0, 0))

# Convert data into a structured numpy array
CVRPTW = []
for entry in data:
    node_data = [
        entry['loc_id'], entry['x_coord'], entry['y_coord'], entry['demand'], entry['service_time'], entry['num_time_windows']
    ] + entry['time_windows']

    CVRPTW.append(node_data)

CVRPTW = np.array(CVRPTW, dtype=object)

# Extract useful information
Nodes = CVRPTW[:, 0]  # nodes
n = len(Nodes)  # number of nodes
Vehicles = {0, 1, 2}

xcoord = CVRPTW[:, 1]  # x coordinates
ycoord = CVRPTW[:, 2]  # y coordinates
q = CVRPTW[:, 3]  # demand
service_duration = CVRPTW[:, 4]  # service time

# Extract time windows for each node
time_windows = CVRPTW[:, 6:]

# Define constants
Q = 200  # vehicle capacity
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

# ---- Decision Variables ----

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

m.update()


# ---- Objective Function ----

obj = quicksum(distance[i,j]*edge[i,j,v] for i in Nodes for j in Nodes for v in Vehicles)
m.setObjective(obj)
m.ModelSense = GRB.MINIMIZE
m.update()

# ---- Constraints ----

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
        quicksum(edge[0,i,v] for i in Nodes if i != 0) <= 1, 'con4[' + str(v) + ']'
    )
    con5[v] = m.addConstr(
        quicksum(edge[i,0,v] for i in Nodes if i != 0) <= 1, 'con5[' + str(v) + ']'
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
                if i != j and j != 0:
                    con10[i,j,v,w] = m.addConstr(
                        service_start[j, v] >= service_start[i, v] + service_duration[int(i)] + distance[i, j] - BIGM * (1 - edge[i, j, v]), 'con10[' + str(i) + ',' + str(j) + ',' + str(v) + ',' + str(w) + ']'
                    )

# ---------- Solve ----------

m.update()
m.write('CVRPTW.lp')
m.Params.TimeLimit = 30600
m.optimize()

# ---------- Print Results ----------

# Extract the routes from the solution
routes = {v: [] for v in Vehicles}
if m.status == GRB.OPTIMAL:
    for v in Vehicles:
        for i in Nodes:
            for j in Nodes:
                if edge[i, j, v].X > 0.5:
                    routes[v].append((i, j))

    # Plot the routes
    plt.figure(figsize=(5, 4))
    plt.scatter(xcoord, ycoord, c='red')
    plt.scatter(xcoord[0], ycoord[0], c='blue', marker='s')  # depot

    for i, (x, y) in enumerate(zip(xcoord, ycoord)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    # get number of vehicles
    num_vehicles = len(Vehicles)

    # Generate distinct colors using a colormap
    cmap = plt.cm.get_cmap('tab20', num_vehicles)  
    colors = [cmap(i) for i in range(num_vehicles)]  

    for v in Vehicles:
        load = 0
        sorted_routes = sorted(routes[v], key=lambda x: service_start[x[0], v].X)
        for (i, j) in sorted_routes:
            load += q[i]
            plt.arrow(xcoord[i], ycoord[i], xcoord[j] - xcoord[i], ycoord[j] - ycoord[i], 
                      color=colors[v], head_width=1.5, length_includes_head=True, label=f'Vehicle {v}' if i == 0 else "")
            # mid_x = (xcoord[i] + xcoord[j]) / 2 
            # mid_y = (ycoord[i] + ycoord[j]) / 2 
            # plt.text(mid_x, mid_y, f'{load}', color='black', fontsize=10, fontweight='bold', ha='center')
            print(f"Vehicle {v} is currently carrying load {load} after visiting node {i}.")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Vehicle Routes with Loads')
    plt.legend()
    plt.show()
else:
    print("No optimal solution found.")


# Create a dataframe to store the results for all vehicles
df_all = pd.DataFrame(columns=['Vehicle', 'Node', 'Service Start', 'Load'])

for v in Vehicles:
    if not routes[v]:
        print(f"Vehicle {v} is not used.")
    else:
        print(f"Vehicle {v} service times and loads:")
        sorted_routes = sorted(routes[v], key=lambda x: service_start[x[0], v].X)
        load = 0
        for (i, j) in sorted_routes:
            load += q[i]
            print(f"  Node {i}: Service start at {service_start[i, v].X:.2f}, Load: {load}")
            new_row = pd.DataFrame({'Vehicle': [v], 'Node': [i], 'Service Start': [service_start[i, v].X], 'Load': [load]})
            if not new_row.isna().all(axis=None):
                df_all = pd.concat([df_all, new_row], ignore_index=True)

        print("\n")

print(df_all)
# show the total distance traveled by all vehicles
total_distance = 0
for v in Vehicles:
    for (i, j) in routes[v]:
        total_distance += distance[i, j]
print(f"Total distance traveled by all vehicles: {total_distance:.2f}")
