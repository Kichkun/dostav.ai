Pickup-and-delivery problem with time windows (PDPTW)

The subdirectory INSTANCES contains the following benchmark instances:

	Li (356 instances)
	Source:
	Li, H., Lim, A.:
	A Metaheuristic for the Pickup and Delivery Problem with Time Windows.
	Proceedings of the 13th IEEE International Conference on Tools with
	Artificial Intelligence:160-167 (2001)

	Ropke (42 instances)
	Source:
	Ropke, S., Pisinger, D.:
	An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery 
	Problem with Time Windows.
	Transp. Sci., 40(4):455-472 (2006)
	
PICKUP_AND_DELIVERY_SECTION :
Each line is of the form

      <integer> <integer> <real> <real> <real> <integer> <integer>
 
The first integer gives the number of the node.
The second integer gives its demand.
The third and fourth number give the earliest and latest time for the node.
The fifth number specifies the service time for the node.
The last two integers are used to specify pickup and delivery. The first of 
these integers gives the index of the pickup sibling, whereas the second integer
gives the index of the delivery sibling.

The subdirectory TOURS contains the best tours found by LKH-3.

Tabulated results can be found in the subdirectory RESULTS.