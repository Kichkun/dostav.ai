import json
import numpy as np

DATA_PATH = '../data/0.json'

json_file = json.load(open(DATA_PATH, 'r'))
output = open('lkh_file.pdptw', 'w')

output.write('NAME : couriers\n')
output.write('TYPE : PDPTW\n')

json_file['orders'] = json_file['orders'][:50]
print(len(json_file['orders']))
dim = 2 * len(json_file['orders'])
output.write(f'DIMENSION : {dim + 1}\n')

output.write(f'VEHICLES : 1\n')
output.write(f'CAPACITY : 10000\n')
output.write('EDGE_WEIGHT_TYPE : MAN_2D\n')
output.write('NODE_COORD_SECTION\n')


inverse_id_map = {}

cnt = 1
for order in json_file['orders']:
    cnt += 1
    inverse_id_map[order['pickup_point_id']] = cnt
    cnt += 1
    inverse_id_map[order['dropoff_point_id']] = cnt

def manhattan(rhs, lhs):
    rhs = np.array(rhs)
    lhs = np.array(lhs)
    return np.abs(rhs - lhs).sum()

def full_matrix(x, y, orders):
    coords = [(x, y, 'start', 0)]

    for order in orders:
        coords.append((order['pickup_location_x'], order['pickup_location_y'], 'pickup', order['payment']))
        coords.append((order['pickup_location_x'], order['pickup_location_y'], 'dropoff', order['payment']))

    dists_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i == j:
                continue

            x_0, y_0, t_0, p_0 = coords[i]
            x_1, y_1, t_1, p_1 = coords[j]
            if t_0 == 'start':
                dists_matrix[i, j] = -2 * manhattan((x_0, x_1), (y_0, y_1))

            if t_1 == 'pickup':
                dists_matrix[i, j] = -2 * manhattan((x_0, x_1), (y_0, y_1))

            if t_1 == 'dropoff':
                dists_matrix[i, j] = p_1 - 2 * manhattan((x_0, x_1), (y_0, y_1))

            if t_1 == 'start':
                dists_matrix[i, j] = -300

    return dists_matrix

cur_x = json_file["couriers"][0]["location_x"]
cur_y = json_file["couriers"][0]["location_y"]

fm = full_matrix(cur_x, cur_y, json_file['orders'])
print(fm)

output.write(f'1 {cur_x} {cur_y}\n')

cnt = 1
for order in json_file['orders']:
    cnt += 1
    output.write(f'{cnt} {order["pickup_location_x"]} {order["pickup_location_y"]}\n')
    cnt += 1
    output.write(f'{cnt} {order["dropoff_location_x"]} {order["dropoff_location_y"]}\n')

output.write('PICKUP_AND_DELIVERY_SECTION\n')

output.write(f'1 0 366 1439 0 0 0\n')
cnt = 1
for order in json_file['orders']:
    cnt += 1
    distance = abs(order['pickup_location_x'] - order['dropoff_location_x']) + abs(order['pickup_location_y'] - order['dropoff_location_y'])
    print(distance)
    output.write(f'{cnt} 1 {order["pickup_from"]} {order["pickup_to"]} {distance + 10} 0 {inverse_id_map[order["dropoff_point_id"]]}\n')
    cnt += 1
    output.write(f'{cnt} -1 {order["dropoff_from"]} {order["dropoff_to"]} 10 {inverse_id_map[order["pickup_point_id"]]} 0\n')

output.write('DEPOT_SECTION\n')
output.write('1\n')
output.write('-1\n')
output.write('EOF')

output.close()
