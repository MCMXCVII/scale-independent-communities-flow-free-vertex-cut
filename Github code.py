import networkx as nx
import matplotlib.pyplot as plt
import itertools


# # # # # # # # # # Communities # # # # # # # # # #

def get_communities(graph, positive_treshold=0, relative_treshold=0.0, negative_treshold=float('inf'),
                    also_subcommunities=False):
    """
    Returns communities with the given properties.
    :param graph: the graph in which the communities are searched.
    :param positive_treshold: the required permissive node connectivity.
    :param relative_treshold: the required relative node connectivity.
    :param negative_treshold: the required restrictive node connectivity.
    :param also_subcommunities: if also subcommunities of the communities should be returned.
    :return: tuples of data and sets of nodes which represent communities.
    """

    def scale_invariant_connectedness(k, v, _):
        return (k >= positive_treshold) and \
               (k >= relative_treshold * (v - 1)) and \
               (k >= v - 1 - negative_treshold)

    def data_generator(k, v, e):
        return k #, k / (v - 1) if v > 1 else 0, v - 1 - k

    return _get_communities(graph, scale_invariant_connectedness, data_generator,
                            also_subcommunities, insufficient_k=positive_treshold - 1)


def _get_communities(graph, property_checker, data_generator, also_subcommunities=False, insufficient_k=-1):
    """
    Returns communities with the given properties.
    :param graph: the graph in which the communities are searched.
    :param property_checker: the function that checks the properties.
    :param data_generator: the function that generates data for the results.
    :param also_subcommunities: if also subcommunities of the communities should be returned.
    :param parent_insufficient_k: optional initial node connectivity that is insufficient for a commmunity.
    :post the graph is unchanged.
    :return: tuples of data and sets of nodes which represent communities.
    """

    if graph.is_directed() or graph.is_multigraph() or graph.number_of_selfloops() > 0:
        raise NotImplementedError('Only undirected graphs without self-loops and parallel edges are supported!')

    # 0. REMOVE DEGREE
    stack = [
        (component, insufficient_k)
        for component in connected_components(graph, {node for node, degree in graph.degree if degree > insufficient_k})
    ]

    # Depth-first search
    while stack:
        node_set, parent_insufficient_k = stack.pop()

        # 1. CALCULATE BASIC PROPERTIES
        subgraph = graph.subgraph(node_set)
        v = subgraph.number_of_nodes()
        e = subgraph.number_of_edges()

        # 2. CALCULATE SUFFICIENT NODE CONNECTIVITY WITH SUFFICIENT NODE CUT
        k, component_a, component_b = sufficient_vertex_cut(subgraph, parent_insufficient_k)

        # 3. STOP IF VALID COMMUNITY OR COMPLETE GRAPH
        if k > parent_insufficient_k and property_checker(k, v, e):  # if valid community
            yield (data_generator(k, v, e), node_set)
            if not also_subcommunities:
                continue
        if e == v * (v - 1) // 2:  # if complete graph
            continue

        # 4. ADD COMPONENTS AND REMOVE DEGREE
        new_insufficient_k = max(k, parent_insufficient_k)
        component_a = {node for node in component_a if subgraph.degree(node) > new_insufficient_k}
        component_b = {node for node in component_b if subgraph.degree(node) > new_insufficient_k}
        if component_a:
            stack.append((component_a, new_insufficient_k))
        if component_b:
            stack.append((component_b, new_insufficient_k))


def sufficient_vertex_cut(graph, sufficient_cut_size):
    """
    If the node connectivity of the graph is bigger than sufficient_cut_size,
    returns the exact node connectivity and a minimal node cut,
    else returns a node connectivity number between the actual node connectivity and sufficient_cut_size
    and a node cuts of size smaller than or equal to sufficient_cut_size.
    The basic idea of this function is to stop the computation of node cuts as fast as possible.
    :param graph: the graph in which the node cuts are calculated.
    :param sufficient_cut_size: node cuts with this size or smaller can always be returned.
    :post the graph is unchanged.
    :return: a tuple of the cut size and the two components
    """

    # Choose a node with minimum degree.
    v = min(graph, key=graph.degree)
    v_neighbors = set(graph[v])

    # Initial node cutset is all neighbors of the node with minimum degree.
    min_component_a = v_neighbors.union({v})
    non_neighbors = set(graph.nodes).difference(min_component_a)
    min_component_b = non_neighbors.union(v_neighbors)
    min_size = len(v_neighbors)

    if min_size <= sufficient_cut_size:
        return min_size, min_component_a, min_component_b

    # Compute st node cuts between v and all its non-neighbors nodes in G.
    for w in non_neighbors:
        this_size, _, this_component, this_other_component = flow_free_vertex_cut(graph, v, w, sufficient_cut_size)
        if min_size > this_size:
            min_size, min_component_a, min_component_b = this_size, this_component, this_other_component
            if min_size <= sufficient_cut_size:
                return min_size, min_component_a, min_component_b

    # Also for non adjacent pairs of neighbors of v.
    for x, y in itertools.combinations(v_neighbors, 2):
        if y in graph[x]:
            continue
        this_size, _, this_component, this_other_component = flow_free_vertex_cut(graph, x, y, sufficient_cut_size)
        if min_size > this_size:
            min_size, min_component_a, min_component_b = this_size, this_component, this_other_component
            if min_size <= sufficient_cut_size:
                return min_size, min_component_a, min_component_b

    return min_size, min_component_a, min_component_b


def flow_free_vertex_cut(graph, s, t, sufficient_cut_size=0):
    """
    Returns a minimum node cut between node s and node t that disconnects the graph.
    :param graph: a graph.
    :param s: a node in the graph.
    :param t: a node in the graph.
    :exception ValueError if s equals t or s is connected to t.
    :return: the length of the cut, the cut and the two resulting components.
    """

    if s == t:
        raise ValueError('Node s (= ' + str(s) + ') cannot equal node t (= ' + str(t) + ')!')
    if graph.has_edge(s, t):
        raise ValueError('Graph cannot have edge between s (= ' + str(s) + ') and t (= ' + str(t) + ')!')
    if graph.number_of_nodes() < 2:
        raise ValueError('Graph must contain at least two nodes!')

    # Prevent changes to graph
    cut_graph = graph.copy()

    # Prepare wave from s
    # A wave is a set of nodes without shared edges, all visited nodes have been removed from the graph.
    s_wave = list(cut_graph[s])
    cut_graph.remove_node(s)
    cut_graph.remove_edges_from(itertools.combinations(s_wave, 2))
    s_wave.sort(key=cut_graph.degree, reverse=True)

    # Prepare wave from t
    # A wave is a set of nodes without shared edges, all visited nodes have been removed from the graph.
    t_wave = list(cut_graph[t])
    cut_graph.remove_node(t)
    cut_graph.remove_edges_from(itertools.combinations(t_wave, 2))
    t_wave.sort(key=cut_graph.degree, reverse=True)

    # Define one of the components of a node cut of the waves
    s_component = {s}
    t_component = {t}

    # Define the intersection of the waves
    # Nodes in the intersection are no longer used in the computation, only in the resulting node cut
    wave_intersection = set()

    # Define function for reuse.
    # Update intersection
    def refresh_intersection():
        nonlocal cut_graph
        nonlocal s_wave
        nonlocal t_wave
        nonlocal wave_intersection
        i = 0
        while i < len(s_wave):
            node = s_wave[i]
            if node in t_wave:
                s_wave.pop(i)
                t_wave.remove(node)
                cut_graph.remove_node(node)
                wave_intersection.add(node)
            else:
                i += 1

    # Update intersection
    refresh_intersection()

    # Initialize the minimum cut variables.
    if len(s_wave) < len(t_wave):
        min_node_cut = wave_intersection.union(s_wave)
        min_component = s_component.copy()
    else:
        min_node_cut = wave_intersection.union(t_wave)
        min_component = t_component.copy()

    # Define function for reuse.
    # Update a wave by removing a node in the wave and adding all new neighbours.
    def remove_node(node, wave, component):
        nonlocal cut_graph
        new_nodes = list(n for n in cut_graph[node] if n not in wave)
        wave.pop()
        wave.extend(new_nodes)
        cut_graph.remove_node(node)
        component.add(node)
        cut_graph.remove_edges_from((u, v) for u in new_nodes for v in wave)
        wave.sort(key=cut_graph.degree, reverse=True)

    # Define function for reuse.
    # Update the minimum cut variables.
    def recheck_min_cut():
        nonlocal s_wave
        nonlocal t_wave
        nonlocal wave_intersection
        nonlocal min_node_cut
        nonlocal min_component
        if len(s_wave) + len(wave_intersection) < len(min_node_cut):
            min_node_cut = wave_intersection.union(s_wave)
            min_component = s_component.copy()
        elif len(t_wave) + len(wave_intersection) < len(min_node_cut):
            min_node_cut = wave_intersection.union(t_wave)
            min_component = t_component.copy()

    # Start update loop
    while s_wave and t_wave and len(min_node_cut) > sufficient_cut_size:

        # Select the node with the lowest degree.
        # Invar: the waves are sorted in descending order.
        s_node = s_wave[-1]
        s_degree = cut_graph.degree(s_node)
        t_node = t_wave[-1]
        t_degree = cut_graph.degree(t_node)

        if s_degree <= t_degree:
            # Update the s_wave by removing a node in the wave and adding all new neighbours.
            remove_node(s_node, s_wave, s_component)
        else:
            # Update the t_wave by removing a node in the wave and adding all new neighbours.
            remove_node(t_node, t_wave, t_component)

        # Update intersection
        refresh_intersection()

        # Update the minimum cut variables.
        recheck_min_cut()

    # Check if graph is disconnected
    if len(min_node_cut) > sufficient_cut_size and not wave_intersection:
        min_component = set(next(connected_components(graph, {s})))
        min_node_cut = {}

    # Calculate the minimum node cut components.
    other_component = set(graph).difference(min_component)
    min_component.update(min_node_cut)

    # Return all data.
    return len(min_node_cut), min_node_cut, min_component, other_component


def connected_components(graph, source_nodes):
    """
    Performs a simple depth first search to find the connected components of the nodes in source_nodes.
    :param graph: the graph with the connection data.
    :param source_nodes: the nodes for which the connected components will be calculated.
    :post the graph is unchanged.
    :return: a generator of sets of nodes that belong to the same connected component.
    """
    source_nodes = set(source_nodes)
    while source_nodes:
        component = set()
        stack = [source_nodes.pop()]
        while stack:
            node = stack.pop()
            if node not in component:
                component.add(node)
                stack.extend(graph[node])
                try:
                    source_nodes.remove(node)
                except KeyError:
                    pass
        yield component


# # # # # # # # # # Visualisations # # # # # # # # # #

def draw_graph(graph, highlight_nodes=None, highlight_edges=None, block=True, color='r', node_size=300, line_width=1.0):
    """
    Draws a NetworkX graph.
    :param graph: the graph to draw
    :param highlight_nodes: optional nodes to highlight
    :param highlight_edges: optional edges to highlight
    :param block: if this function should block until the user closes the window
    :param color: the color of the highlighted parts
    """
    if highlight_nodes is not None:
        node_colors = [color if v in highlight_nodes else 'grey' for v in graph.nodes()]
    else:
        node_colors = 'grey'
    if highlight_edges is not None:
        edge_colors = [color
                       if (u, v) in highlight_edges or (v, u) in highlight_edges else 'grey'
                       for (u, v) in graph.edges()]
    else:
        edge_colors = 'grey'
    nx.drawing.draw_kamada_kawai(
        graph,
        with_labels=True,
        node_size=node_size,
        node_color=node_colors,
        width=line_width,
        edge_color=edge_colors,
        font_color='black'
    )
    plt.draw()
    print("Showing figure...")
    plt.show(block)
    print("Figure closed!")
