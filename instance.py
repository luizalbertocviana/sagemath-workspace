from sage.all import Graph, DiGraph

def read_graph(filename: str) -> Graph:
    g_file = open(filename)

    first_line = g_file.readline()
    num_verts = int(first_line)

    g = Graph(num_verts)

    for line in g_file:
        u, v, w = map(int, line.split())
        g.add_edge(u, v, w)

    return g

def read_digraph(filename: str) -> DiGraph:
    d_file = open(filename)

    first_line = d_file.readline()
    num_verts = int(first_line)

    d = DiGraph(num_verts)

    for line in d_file:
        u, v = map(int, line.split())
        d.add_edge(u, v)

    return d

def read_dep_bounds(filename: str) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    lb_dep = dict()
    ub_dep = dict()

    b_file = open(filename)

    for line in b_file:
        u, v, lb, ub = map(int, line.split())
        lb_dep[u, v] = lb
        ub_dep[u, v] = ub

    return (lb_dep, ub_dep)

def read_instance(g_filename: str, d_filename: str, b_filename: str) -> instance:
    g = read_graph(g_filename)
    d = read_digraph(d_filename)
    lb_dep, ub_dep = read_dep_bounds(b_filename)

    inst = instance()

    inst.graph = g
    inst.digraph = d
    inst.lb_dep = lb_dep
    inst.ub_dep = ub_dep

    return inst
