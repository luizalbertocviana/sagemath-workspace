import multiprocessing as mp
from os import mkdir
from os.path import exists

from typing import Callable, List, Tuple
from cplex import Cplex
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

import pandas as pd

from .instance import instance, read_instance
from .model import create_model

class logging_filenames:
    solution: str
    solving: str

class solving_parameters:
    memory_limit: int
    time_limit: int

class solving_info:
    sol: SolveSolution
    time: float
    gap: float
    best_int_solution: float
    best_upper_bound: float
    number_nodes: int
    number_iterat: int
    status: str

def solve_model(model: Model, parameters: solving_parameters, filenames: logging_filenames) -> solving_info:
    model.quality_metrics = True

    cplex: Cplex = model.get_cplex()
    cplex.parameters.mip.limits.treememory.set(parameters.memory_limit)
    cplex.parameters.timelimit.set(parameters.time_limit)

    solution: SolveSolution = model.solve(log_output=filenames.solving)
    solution.export_as_mst(path=filenames.solution)

    info = solving_info()

    info.sol               = solution
    info.time              = solution.solve_details.time
    info.gap               = solution.solve_details.mip_relative_gap * 100
    info.best_int_solution = solution.objective_value
    info.best_upper_bound  = solution.solve_details.best_bound
    info.number_nodes      = solution.solve_details.nb_nodes_processed
    info.number_iterat     = solution.solve_details.nb_iterations
    info.status            = solution.solve_details.status
    #TODO include root gap

    return info

def solve_instances(instance_ids: List[str],
                    instance_getter: Callable[[str], instance],
                    instance_filename_getter: Callable[[str], logging_filenames],
                    parameters: solving_parameters,
                    action: Callable[[str, solving_info], None]) -> None:
    def solve_model_subprocess(model: Model, parameters: solving_parameters, filenames: logging_filenames, q: mp.Queue) -> None:
        info = solve_model(model, parameters, filenames)

        q.put(info)

    queue: mp.Queue[solving_info] = mp.Queue()

    for instance_id in instance_ids:
        instance  = instance_getter(instance_id)
        filenames = instance_filename_getter(instance_id)
        model     = create_model(instance)

        subprocess = mp.Process(target=solve_model_subprocess, args=(model, parameters, filenames, queue))
        subprocess.start()
        subprocess.join()

        info = queue.get()

        action(instance_id, info)

def create_table() -> pd.DataFrame:
    cols = ['time', 'gap', 'best_int_solution', 'best_upper_bound', 'number_nodes', 'number_iterat', 'status']
    df = pd.DataFrame(columns=cols)

    return df

def save_table(table: pd.DataFrame, filename: str) -> None:
    table.to_csv(filename)

def load_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def register_instance_info(table: pd.DataFrame, instance_id: str, info: solving_info) -> None:
    line = (info.time, info.gap,info.best_int_solution, info.best_upper_bound, info.number_nodes, info.number_iterat, info.status)

    table.loc[instance_id] = line

def solve_instances_write_table(instance_ids: List[str],
                                table_filename: str,
                                instance_getter: Callable[[str], instance],
                                instance_filename_getter: Callable[[str], logging_filenames],
                                parameters: solving_parameters):
    if exists(table_filename):
        table = load_table(table_filename)
    else:
        table = create_table()

    def register_into_table(instance_id: str, info: solving_info):
        register_instance_info(table, instance_id, info)

        save_table(table, table_filename)

    solve_instances(instance_ids, instance_getter, instance_filename_getter, parameters, register_into_table)

