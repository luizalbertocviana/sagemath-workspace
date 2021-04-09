from typing import Callable

from sage.all import Graph, DiGraph

from docplex.mp.solution import SolveSolution
from docplex.mp.model import Model

class instance:
    graph: Graph
    root: int
    digraph: DiGraph
    lb_dep: dict
    ub_dep: dict

def create_model(inst: instance) -> Model:
    g      = inst.graph
    d      = inst.digraph
    root   = inst.root
    lb_dep = inst.lb_dep
    ub_dep = inst.ub_dep

    model = Model()

    # number of vertices
    n = g.num_verts()

    # establishes relations (back and forth) between the edges of g and the vertices
    # of d, that is, edge e of g corresponds to vertex e_to_index[e] of d
    e_to_index = dict()
    index_to_e = dict()
    for (idx, (u, v)) in enumerate(g.edge_iterator(labels=False)):
        e_to_index[u, v] = idx
        index_to_e[idx] = (u, v)

    # model variables:
    x = dict() # binary variables, one for each edge
    y = dict() # binary variables, for each edge uv we have y_uv and y_vu
    l = dict() # nonnegative continuous variables, one for each vertex

    # creating x and y variables
    for (u, v) in g.edge_iterator(labels=False):
        x[u, v] = model.binary_var(name="x_%d_%d" % (u, v))

        y[u, v] = model.binary_var(name="y_%d_%d" % (u, v))
        y[v, u] = model.binary_var(name="y_%d_%d" % (v, u))

    # creating l variables
    for v in g.vertex_iterator():
        l[v] = model.continuous_var(lb=g.distance(root, v) + 1, ub=n, name="l_%d" % v)

    # model constraints

    # x_uv = y_uv + y_vu
    for (u, v) in g.edge_iterator(labels=False):
        model.add_constraint(x[u, v] == y[u, v] + y[v, u])

    # \sum_{uv \in E(G)} (y_uv + y_vu) = |V(G)| - 1
    model.add_constraint(sum(var for var in y.values()) == n - 1)

    # \sum_{u \in N(v)} y_uv = 1
    for v in g.vertex_iterator():
        if v != root:
            model.add_constraint(sum(y[u, v] for u in g.neighbor_iterator(v)) == 1)

    # \sum_{u \in N(r)} y_ur = 0
    model.add_constraint(sum(y[u, root] for u in g.neighbor_iterator(root)) == 0)

    # l_r = 1
    model.add_constraint(l[root] == 1)

    # l_u - l_v + 1 <= (|V(G)| - d(r, v))(1 - y_uv)
    for (u, v) in g.edge_iterator(labels=False):
        if v != root:
            model.add_constraint(l[u] - l[v] + 1 <= (n - g.distance(root, v))(1 - y[u, v]))
        if u != root:
            model.add_constraint(l[v] - l[u] + 1 <= (n - g.distance(root, u))(1 - y[v, u]))

    # dependency constraints
    for (u, v) in g.edge_iterator(labels=False):
        # \sum_{e2 \in dep(e1)} x_e2 >= l_e1x_e1
        model.add_constraint(
            sum(x[index_to_e[dep_idx]] for dep_idx in d.neighbor_in_iterator(e_to_index[u, v]))
            >= lb_dep[u, v] * x[u, v])
        # \sum_{e2 \in dep(e1)} x_e2 <= |dep(e1)| - (|dep(e1)| - u(e1))x_e1 
        num_deps = d.in_degree(e_to_index[u, v])
        model.add_constraint(
            sum(x[index_to_e[dep_idx]] for dep_idx in d.neighbor_in_iterator(e_to_index[u, v]))
            <= num_deps - (num_deps - ub_dep[u, v]) * x[u, v] )

    # objective function

    model.minimize(sum(w * x[u, v] for (u, v, w) in g.edge_iterator()))

    return model

class solving_info:
    sol: SolveSolution
    time: float
    gap: float
    best_int_solution: float
    best_upper_bound: float
    number_nodes: int
    number_iterat: int
    status: str

class solving_parameters:
    memory_limit: int
    time_limit: int

class logging_filenames:
    solution: str
    solving: str

def solve_model(model: Model, parameters: solving_parameters, filenames: logging_filenames) -> solving_info:
    model.quality_metrics = True

    cplex = model.get_cplex()
    cplex.parameters.mip.limits.treememory.sett(parameters.memory_limit)
    cplex.parameters.timelimit.set9(parameters.time_limit)

    solution = model.solve(log_output=filenames.solving)
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

    return info

def solve_instances(instance_ids: list[str],
                    instance_getter: Callable[[str], instance],
                    instance_filename_getter: Callable[[str], logging_filenames],
                    parameters: solving_parameters,
                    action: Callable[[str, solving_info], None]) -> None:
    for instance_id in instance_ids:
        instance = instance_getter(instance_id)
        filenames = instance_filename_getter(instance_id)
        model = create_model(instance)
        info = solve_model(model, parameters, filenames)

        action(instance_id, info)
