import json

DATA_PATH = '../data/0.json'

json_file = json.load(open(DATA_PATH, 'r'))
output = open('lkh_file_all.pdptw', 'w')

output.write('NAME : couriers\n')
output.write('TYPE : PDPTW\n')

json_file['orders'] = json_file['orders'][:50]
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



output.write(f'1 {json_file["couriers"][0]["location_x"]} {json_file["couriers"][0]["location_y"]}\n')

cnt = 1
for order in json_file['orders']:
    cnt += 1
    output.write(f'{cnt} {order["pickup_location_x"]} {order["pickup_location_y"]}\n')
    cnt += 1
    output.write(f'{cnt} {order["dropoff_location_x"]} {order["dropoff_location_y"]}\n')

output.write('PICKUP_AND_DELIVERY_SECTION\n')

output.write(f'1 0 360 2000 0 0 0\n')
cnt = 1
for order in json_file['orders']:
    cnt += 1
    distance = abs(order['pickup_location_x'] - order['dropoff_location_x']) + abs(order['pickup_location_y'] - order['dropoff_location_y'])
    print(distance)
    output.write(f'{cnt} 20 {order["pickup_from"]} {order["pickup_to"]} 20 0 {inverse_id_map[order["dropoff_point_id"]]}\n')
    cnt += 1
    output.write(f'{cnt} -20 {order["dropoff_from"]} {order["dropoff_to"]} 20 {inverse_id_map[order["pickup_point_id"]]} 0\n')

output.write('DEPOT_SECTION\n')
output.write('1\n')
output.write('-1\n')
output.write('EOF')

output.close()
