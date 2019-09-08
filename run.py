from __future__ import print_function
import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm_notebook
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

data = load_data("../data/hard_input.json")

def load_data(file):
    """Загрузка входных данных из файла"""
    with open(file, 'r') as f:
        input_data = json.load(f)
    couriers = {}
    orders = {}
    points = {}
    for depotData in input_data['depots']:
        points[depotData['point_id']] = {
            'location': [depotData['location_x'], depotData['location_y']],
            'timewindow': [0, 1439],
        }
    for courierData in input_data['couriers']:
        couriers[courierData['courier_id']] = {
            'location': [courierData['location_x'], courierData['location_y']],
            'time': 360,
        }
    for orderData in input_data['orders']:
        points[orderData['pickup_point_id']] = {
            'location': [orderData['pickup_location_x'], orderData['pickup_location_y']],
            'timewindow': [orderData['pickup_from'], orderData['pickup_to']],
            'order_time': {orderData['order_id']: orderData['pickup_from']}
        }
        points[orderData['dropoff_point_id']] = {
            'location': [orderData['dropoff_location_x'], orderData['dropoff_location_y']],
            'timewindow': [orderData['dropoff_from'], orderData['dropoff_to']],
        }
        orders[orderData['order_id']] = orderData
    return couriers, orders, points

def get_travel_duration_minutes(location1, location2):
    """Время перемещения курьера от точки location1 до точки location2 вминутах"""
    distance = abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
    return 10 + distance

def get_current_courier_reward(time, payment):
    work_payment = time * 2
    profit = orders_payment - work_payment
    return profit

def delete_element(array, idx):
    result = []
    for i in range(len(array)):
        if not(i == idx):
            result.append(array[i])
    return result
from tqdm import tqdm_notebook

def get_by_id(point_id):
    if point_id < 10000:
        return couriers_start_position[point_id]
    if point_id < 60000:
        return points_pickup[point_id-40001]
    else:
        return points_dropoff[point_id-60001]

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector 


def get_directions_table(pickup_ids):
    directions_table = []
    for pickup_id in pickup_ids:
        n = pickup_id - 40001
        n_pickup_in_table = len(data[0]) + n
        n_dropoff_in_table = len(data[0]) + len(data[1]) + n 
        directions_table.append([n_pickup_in_table, n_dropoff_in_table])
    return directions_table

def get_time_windows(pickup_id, dropoff_id, n_couriers):
    tmp_string = np.concatenate((np.array(range(n_couriers)), 
                                 pickup_id, 
                                 dropoff_id))
    tmp_couriers = [(360,1440) for i in range(n_couriers)]
    tmp_pickup = []
    tmp_dropoff = []
    for i in tqdm_notebook(range(len(pickup_id))):
        tmp_pickup.append((data[1][pickup_id[i] - 30000]['pickup_from'], 
                           data[1][pickup_id[i]-30000]['pickup_to']))
        tmp_dropoff.append((data[1][dropoff_id[i] - 50000]['dropoff_from'], 
                           data[1][dropoff_id[i]-50000]['dropoff_to']))
    result = []
    for i in tqdm_notebook(range(len(tmp_couriers))):
        result.append(tmp_couriers[i])
    for i in range(len(tmp_pickup)):
        result.append(tmp_pickup[i])
    for i in range(len(tmp_dropoff)):
        result.append(tmp_dropoff[i])
    return result
    
def get_time_windows(pickup_id, dropoff_id, n_couriers):
    tmp_string = np.concatenate((np.array(range(n_couriers)), 
                                 pickup_id, 
                                 dropoff_id))
    tmp_couriers = [(0,300) for i in range(n_couriers)]
    tmp_pickup = []
    tmp_dropoff = []
    for i in tqdm_notebook(range(len(pickup_id))):
        tmp_pickup.append((data[1][pickup_id[i] - 30000]['pickup_from'], 
                           data[1][pickup_id[i]-30000]['pickup_to']))
        tmp_dropoff.append((data[1][dropoff_id[i] - 50000]['dropoff_from'], 
                           data[1][dropoff_id[i]-50000]['dropoff_to']))
    result = []
    for i in tqdm_notebook(range(len(tmp_couriers))):
        result.append(tmp_couriers[i])
    for i in range(len(tmp_pickup)):
        result.append(tmp_pickup[i])
    for i in range(len(tmp_dropoff)):
        result.append(tmp_dropoff[i])
    return result


def create_distance_matrix(pickup_id, dropoff_id, n_couriers):
    tmp_string = np.concatenate((np.array(range(n_couriers)), 
                                 pickup_id, 
                                 dropoff_id))
    n_pickup = len(data[1])
    n_dropoff = len(data[1])
    
    
    print(pickup_id, dropoff_id)
    
    matrix = []
    for i in tqdm_notebook(range(len(tmp_string))):
        tmp = []
        for j in range(len(tmp_string)):
            tmp.append(get_travel_duration_minutes(get_by_id(tmp_string[j]), 
                                                   get_by_id(tmp_string[i])))
            
            
        matrix.append(tmp)
        
    result = np.pad(matrix, 1, pad_with)[:-1,:-1]
    
    matrix_map = {}
    matrix_map[0] = (None, None, 'base')
    for i in range(n_couriers):
        matrix_map[i + 1] = (None, None, 'courier')
        
    for i in range(n_pickup):
        matrix_map[i + 1 + n_couriers] = (pickup_id[i], pickup_id[i] - 30000, 'pickup')
        
    for i in range(n_dropoff):
        matrix_map[i + 1 + n_couriers + n_pickup] = (dropoff_id[i], dropoff_id[i] - 50000, 'dropoff')
        
    return result, matrix_map
data = load_data("../data/hard_input.json")
couriers_start_position = [data[0][i+1]['location'] for i in range(len(data[0]))]
points_pickup = np.array([[data[1][i+10001]['pickup_location_x'],
                           data[1][i+10001]['pickup_location_y']] for i in range(len(data[1]))])
time_pickup = np.array([[data[1][10001+i]['pickup_from'],
                         data[1][10001+i]['pickup_to']] for i in range(len(data[1]))])
time_dropoff = np.array([[data[1][10001+i]['dropoff_from'],
                          data[1][10001+i]['dropoff_to']] for i in range(len(data[1]))])    
points_dropoff = np.array([[data[1][i+10001]['dropoff_location_x'],
                            data[1][i+10001]['dropoff_location_y']] for i in range(len(data[1]))])
payments = np.array([data[1][10001+i]['payment'] for i in range(len(data[1]))])
pickup_id = np.array([data[1][10001+i]['pickup_point_id'] for i in range(len(data[1]))])
dropoff_id = np.array([data[1][10001+i]['dropoff_point_id'] for i in range(len(data[1]))])
from tqdm import tqdm_notebook

def get_by_id(point_id):
    if point_id < 10000:
        return couriers_start_position[point_id]
    if point_id < 60000:
        return points_pickup[point_id-40001]
    else:
        return points_dropoff[point_id-60001]

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector 


def get_directions_table(pickup_ids):
    directions_table = []
    for pickup_id in pickup_ids:
        n = pickup_id - 40001
        n_pickup_in_table = len(data[0]) + n
        n_dropoff_in_table = len(data[0]) + len(data[1]) + n 
        directions_table.append([n_pickup_in_table, n_dropoff_in_table])
    return directions_table

def get_time_windows(pickup_id, dropoff_id, n_couriers):
    tmp_string = np.concatenate((np.array(range(n_couriers)), 
                                 pickup_id, 
                                 dropoff_id))
    tmp_couriers = [(360,1440) for i in range(n_couriers)]
    tmp_pickup = []
    tmp_dropoff = []
    for i in tqdm_notebook(range(len(pickup_id))):
        tmp_pickup.append((data[1][pickup_id[i] - 30000]['pickup_from'], 
                           data[1][pickup_id[i]-30000]['pickup_to']))
        tmp_dropoff.append((data[1][dropoff_id[i] - 50000]['dropoff_from'], 
                           data[1][dropoff_id[i]-50000]['dropoff_to']))
    result = []
    for i in tqdm_notebook(range(len(tmp_couriers))):
        result.append(tmp_couriers[i])
    for i in range(len(tmp_pickup)):
        result.append(tmp_pickup[i])
    for i in range(len(tmp_dropoff)):
        result.append(tmp_dropoff[i])
    return result
    
def get_time_windows(pickup_id, dropoff_id, n_couriers):
    tmp_string = np.concatenate((np.array(range(n_couriers)), 
                                 pickup_id, 
                                 dropoff_id))
    tmp_couriers = [(0,300) for i in range(n_couriers)]
    tmp_pickup = []
    tmp_dropoff = []
    for i in tqdm_notebook(range(len(pickup_id))):
        tmp_pickup.append((data[1][pickup_id[i] - 30000]['pickup_from'], 
                           data[1][pickup_id[i]-30000]['pickup_to']))
        tmp_dropoff.append((data[1][dropoff_id[i] - 50000]['dropoff_from'], 
                           data[1][dropoff_id[i]-50000]['dropoff_to']))
    result = []
    for i in tqdm_notebook(range(len(tmp_couriers))):
        result.append(tmp_couriers[i])
    for i in range(len(tmp_pickup)):
        result.append(tmp_pickup[i])
    for i in range(len(tmp_dropoff)):
        result.append(tmp_dropoff[i])
    return result


def create_distance_matrix(pickup_id, dropoff_id, n_couriers):
    tmp_string = np.concatenate((np.array(range(n_couriers)), 
                                 pickup_id, 
                                 dropoff_id))
    n_pickup = len(data[1])
    n_dropoff = len(data[1])
    
    
    print(pickup_id, dropoff_id)
    
    matrix = []
    for i in tqdm_notebook(range(len(tmp_string))):
        tmp = []
        for j in range(len(tmp_string)):
            tmp.append(get_travel_duration_minutes(get_by_id(tmp_string[j]), 
                                                   get_by_id(tmp_string[i])))
            
            
        matrix.append(tmp)
        
    result = np.pad(matrix, 1, pad_with)[:-1,:-1]
    
    matrix_map = {}
    matrix_map[0] = (None, None, 'base')
    for i in range(n_couriers):
        matrix_map[i + 1] = (None, None, 'courier')
        
    for i in range(n_pickup):
        matrix_map[i + 1 + n_couriers] = (pickup_id[i], pickup_id[i] - 30000, 'pickup')
        
    for i in range(n_dropoff):
        matrix_map[i + 1 + n_couriers + n_pickup] = (dropoff_id[i], dropoff_id[i] - 50000, 'dropoff')
        
    return result, matrix_map
time_windows = get_time_windows(pickup_id, dropoff_id, n_couriers=len(data[0]))
dist=get_directions_table(pickup_id)
matrix, matrix_map = create_distance_matrix(pickup_id, dropoff_id, len(data[0]))
n_couriers = len(data[0])
def create_data_model(matrix, dist, n_couriers, time_windows):
    d = {'distance_matrix': matrix, 
         'pickups_deliveries':dist,
        'num_vehicles':n_couriers,
        'depot':0,
         'time_windows':time_windows}
    return d
    # [END data_model]
ss = create_data_model(matrix, dist, n_couriers, time_windows)
def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        total_distance += route_distance
    print('Total Distance of all routes: {}m'.format(total_distance))
    # [END solution_printer]

def get_solution(data, manager, routing, assignment):
    solutions = {}
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        solutions[vehicle_id] = []
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            solutions[vehicle_id].append(manager.IndexToNode(index))
#            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
#        plan_output += '{}\n'.format(manager.IndexToNode(index))
#        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
#        print(plan_output)
        total_distance += route_distance
    print('Total Distance of all routes: {}m'.format(total_distance))
    return solutions
    # [END solution_printer]
    
def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]
    data = create_data_model(matrix, dist, n_couriers, time_windows)
    # [END data]

    # Create the routing index manager.
    # [START index_manager]
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    # [END index_manager]

    # Create Routing Model.
    # [START routing_model]
    routing = pywrapcp.RoutingModel(manager)

    # [END routing_model]

    # Define cost of each arc.
    # [START arc_cost]
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # [END arc_cost]

    # Add Distance constraint.
    # [START distance_constraint]
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        1440,  # no slack
        500,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    # [END distance_constraint]
    
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        300,  # allow waiting time
        3000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        print(time_window[0], time_window[1])
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))
        
    # Define Transportation Requests.
    # [START pickup_delivery_constraint]
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))
    # [END pickup_delivery_constraint]

    # Setting first solution heuristic.
    # [START parameters]
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    # [END parameters]

    # Solve the problem.
    # [START solve]
    assignment = routing.SolveWithParameters(search_parameters)
    # [END solve]

    # Print solution on console.
    # [START print_solution]
    if assignment:
#        print_solution(data, manager, routing, assignment)
        return get_solution(data, manager, routing, assignment), data
    # [END print_solution]

n_couriers = len(data[0])
solution, tmp_data = main()
# [END program]
def create_data_model(matrix, dist, n_couriers, time_windows):
    d = {'distance_matrix': matrix, 
         'pickups_deliveries':dist,
        'num_vehicles':n_couriers,
        'depot':0,
         'time_windows':time_windows}
    return d
    # [END data_model]

def solution_to_json(data, solution, matrix_map):
    result = []
    for sol in solution:
        courier_run = {'courier_id': sol + 1}

        for step in solution[sol]:
            step = int(step)
            if step > int(data['num_vehicles']):
                point_id, order_id, order_type = matrix_map[step]
                courier_run = {'courier_id': int(sol + 1),
                               'order_id': int(order_id),
                               'point_id': int(point_id),
                               'action': order_type}

                result.append(courier_run)


        return result
solution_json = solution_to_json(tmp_data, solution, matrix_map)
json.dump(solution_json, open('solution.json', 'w'))
