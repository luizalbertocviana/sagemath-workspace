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

    if not exists(filenames.solution):
        mkdir(filenames.solution)

    solution.export_as_mst(path=filenames.solution)

    info = solving_info()

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
    def solve_model_subprocess(model: Model, parameters: solving_parameters, filenames: logging_filenames, return_dict) -> None:
        info = solve_model(model, parameters, filenames)

        return_dict['info'] = info

    manager = mp.Manager()
    return_dict = manager.dict()

    for instance_id in instance_ids:
        instance  = instance_getter(instance_id)
        filenames = instance_filename_getter(instance_id)
        model     = create_model(instance)

        subprocess = mp.Process(target=solve_model_subprocess, args=(model, parameters, filenames, return_dict))
        subprocess.start()
        subprocess.join()

        info = return_dict['info']

        action(instance_id, info)

def create_table() -> pd.DataFrame:
    cols = ['time', 'gap', 'best_int_solution', 'best_upper_bound', 'number_nodes', 'number_iterat', 'status']
    df = pd.DataFrame(columns=cols)

    return df

def save_table(table: pd.DataFrame, filename: str) -> None:
    table.to_csv(filename)

def load_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, index_col=0)

def register_instance_info(table: pd.DataFrame, instance_id: str, info: solving_info) -> None:
    line = {"time"              : info.time,
            "gap"               : info.gap,
            "best_int_solution" : info.best_int_solution,
            "best_upper_bound"  : info.best_upper_bound,
            "number_nodes"      : info.number_nodes,
            "number_iterat"     : info.number_iterat,
            "status"            : info.status}

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

    def register_into_table(instance_id: str, info: solving_info) -> None:
        register_instance_info(table, instance_id, info)

        save_table(table, table_filename)

    def unsolved(instance_id: str) -> bool:
        return instance_id not in table.index

    unsolved_instance_ids = list(filter(unsolved, instance_ids))

    solve_instances(unsolved_instance_ids, instance_getter, instance_filename_getter, parameters, register_into_table)

def get_instance_filenames(instance_id: str) -> Tuple[str, str, str]:
    graph_filename: str   = "G_" + instance_id
    digraph_filename: str = "D_" + instance_id
    bounds_filename: str  = "B_" + instance_id

    return (graph_filename, digraph_filename, bounds_filename)

def get_instance_from_id(instance_id: str) -> instance:
    graph_filename, digraph_filename, bounds_filename = get_instance_filenames(instance_id)

    return read_instance(graph_filename, digraph_filename, bounds_filename)

def get_instance_from_directory_and_id(directory: str, instance_id: str) -> instance:
    filenames = get_instance_filenames(instance_id)

    complete_filenames = [directory + "/" + filename for filename in filenames]

    return read_instance(complete_filenames[0], complete_filenames[1], complete_filenames[2])

def get_log_filenames_from_id(instance_id: str) -> logging_filenames:
    filenames = logging_filenames()

    filenames.solution = "SOL_" + instance_id
    filenames.solving = "LOG_" + instance_id

    return filenames

def solve_instances_directory(directory: str, parameters: solving_parameters, output_dir = "solving") -> None:
    instance_id_filename = "instance_ids"

    module_instance_ids_name = directory + "." + instance_id_filename

    module_instance_ids = __import__(module_instance_ids_name)

    instance_ids: List[str] = module_instance_ids.instance_ids.get_instance_ids()

    if not exists(output_dir):
        mkdir(directory + "/" + output_dir)

    table_filename = directory + "/" + output_dir + "/" + "results.csv"

    def get_instance(instance_id: str) -> instance:
        return get_instance_from_directory_and_id(directory, instance_id)

    def get_filenames(instance_id: str) -> logging_filenames:
        lf = get_log_filenames_from_id(instance_id)

        lf.solution = directory + "/" + output_dir + "/" + lf.solution
        lf.solving = directory + "/" + output_dir + "/" + lf.solving

        return lf

    solve_instances_write_table(instance_ids, table_filename, get_instance, get_filenames, parameters)

def solve_instances_directories(directories: List[str]) -> None:
    pars = solving_parameters()

    # megabytes
    pars.memory_limit = 32 * 1024 # 32 GB
    # seconds
    pars.time_limit = 2 * 60 * 60 # 2 hours

    for directory in directories:
        solve_instances_directory(directory, pars)

def solve_instances() -> None:
    num_generators = 20

    dirs = ["generator%d" % i for i in range(1, num_generators + 1)]

    solve_instances_directories(dirs)
