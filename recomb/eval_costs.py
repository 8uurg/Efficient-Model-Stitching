import torchinfo

def embed_cost_stats_in_model(s: torchinfo.ModelStatistics):
    # First, set some variables
    for e in s.summary_list:
        e.seen = False
        e.module.count = 0
        e.module.total_macs = 0
        e.module.total_param_bytes = 0
        e.module.total_output_bytes = 0
        e.module.total_all_bytes = 0

    compute_tree_metrics(s.summary_list[0], root=True)

def compute_tree_metrics(n, root=False):
    total_macs = 0
    total_param_bytes = 0
    total_output_bytes = 0

    # An unexecuted node does nothing.
    if not n.executed:
        return total_macs, total_param_bytes, total_output_bytes

    # Leaf nodes do calculations (presumably?)
    if n.is_leaf_layer:
        total_macs += n.macs
        total_param_bytes += n.param_bytes
        if n.num_params > 0:
            total_output_bytes += n.output_bytes * 2

    if root:
        n.module.direct = []

    # For torchinfo, children should be named 'descendants' rather than 'children'
    # as it includes grandchildren, and further descendants, too. We assume there
    # is an extra attribute that is set to false upon our visit, but set to true
    # once visited.
    for ch in n.children:
        if ch.seen:
            # Was not a direct descendant - skip.
            continue

        if root:
            n.module.direct.append(ch.module)

        ch_total_macs, \
        ch_total_param_bytes, \
        ch_total_output_bytes = compute_tree_metrics(ch)
        
        ch.seen = n

        total_macs += ch_total_macs
        total_param_bytes += ch_total_param_bytes
        total_output_bytes += ch_total_output_bytes

    # This is a call to a module! :)
    n.module.count += 1
    n.module.total_macs += total_macs
    n.module.total_param_bytes += total_param_bytes
    n.module.total_output_bytes += total_output_bytes
    n.module.total_all_bytes += total_param_bytes + total_output_bytes

    return total_macs, total_param_bytes, total_output_bytes

# Reuse stored values
def evaluate_cost_using_cache(net):
    macs_total = 0
    bytes_total = 0

    from collections import Counter
    c = Counter(net.graph.vs["module"])

    for midx, m in enumerate(net.submodules):
        macs_total += m.total_macs * c[midx]
        bytes_total += m.total_all_bytes * c[midx]
    # print(f"macs: {macs_total}; bytes: {bytes_total}")
    return macs_total, bytes_total

def evaluate_cost_torchinfo(net, X):
    s = torchinfo.summary(net, input_data=[X])
    # print(f"macs: {s.total_mult_adds}; bytes: {s.total_output_bytes + s.total_param_bytes}")
    return s.total_mult_adds, s.total_output_bytes + s.total_param_bytes

def compute_minimum_mads_using_cache(net):
    # Create a copy of the graph
    g = net.graph.copy()

    # For every edge, assign a cost proportional to the number of multiply-adds
    def get_module_cost(module_id):
        if module_id < 0:
            return 0
        
        module = net.submodules[module_id]
        try:
            return module.total_macs
        except:
            print(f"module {type(module)} does not have mads attr")
            return 0

    g.es["cost"] = [get_module_cost(g.vs[e.target]["module"]) for e in g.es]
    return g.distances(source=[0], target=[1], weights="cost")[0][0]

def compute_minimum_bytes_using_cache(net):
    # Note - this does assume that we do not reuse memory for layers
    # with the same module ID, which isn't necessarily the case...
    # Create a copy of the graph
    g = net.graph.copy()

    # For every edge, assign a cost proportional to the number of multiply-adds
    def get_module_cost(module_id):
        if module_id < 0:
            return 0
        
        module = net.submodules[module_id]
        try:
            return module.total_all_bytes
        except:
            print(f"module {type(module)} does not have total_all_bytes attr")
            return 0

    g.es["cost"] = [get_module_cost(g.vs[e.target]["module"]) for e in g.es]
    return g.distances(source=[0], target=[1], weights="cost")[0][0]

evaluate_compute_cost = evaluate_cost_using_cache