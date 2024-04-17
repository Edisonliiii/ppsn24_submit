import os
import random

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB

# Get the root directory of your project
project_root = os.path.dirname(os.path.abspath(__file__))
env_var = os.environ


# fix seed
seed = int(os.environ["EXP_SEED"])
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(seed)


## CPO & MIP solver parameters
TIME_LIMIT = 600
NUM_THREAD = 1
LOG_LEVEL = 0  # 0 - no log, 1 - show log
# PRESOLVE_LEVEL = 1  # 1- conservative

# gurobi
_BOUND = 2**31 - 1


# gp.setParam("Presolve", PRESOLVE_LEVEL)
gp.setParam("Seed", seed % _BOUND)
gp.setParam("TimeLimit", TIME_LIMIT)
gp.setParam(GRB.Param.Threads, NUM_THREAD)
gp.setParam("LogToConsole", LOG_LEVEL)


# config
# tw_upper = 5
# lb_threshold = 500
tm_generations = 3000
ea_generations = 1000000
