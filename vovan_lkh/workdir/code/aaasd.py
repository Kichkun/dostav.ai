PAR_BASE = """
PROBLEM_FILE = cities.tsp

OUTPUT_TOUR_FILE = cities.tour.1.{i}
SEED = {i}
TRACE_LEVEL = 1

PATCHING_C = 3
PATCHING_A = 2
CANDIDATE_SET_TYPE = POPMUSIC
INITIAL_PERIOD = 300
MOVE_TYPE = 6
MAX_CANDIDATES = 4
RUNS = 1000
"""

PARS = [ # invidual custom parameters
    # no. 0 generates parameters for rust
    """
    CANDIDATE_FILE = cities.cand
    PI_FILE = cities.pi
    MAX_CANDIDATES = 8
    MOVE_TYPE = 5
    PATCHING_C = 0
    PATCHING_A = 0
    """,
    "SEED = 41849",
    "SEED = 474747503", # seems to be a good seed
    "SEED = 47" # seems to be a good seed
]

for i, par in enumerate(PARS):
    with open("cities.par.1.%d" % i, "w") as f:
        print(PAR_BASE.format(i=i), file=f)
        print(par, file=f)