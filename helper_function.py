import numpy as np
import cobra
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import networkx as nx
import re
import matplotlib as mpl
from scipy import stats
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigs
plt.rcParams['font.family'] = 'Arial'

shared_medium = {
    'EX_mn2(e)': 1000.0, 'EX_cu2(e)': 1000.0, 'EX_pi(e)': 1000.0,
    'EX_cobalt2(e)': 1000.0, 'EX_h(e)': 1000.0, 'EX_mg2(e)': 1000.0,
    'EX_co2(e)': 1000.0, 'EX_o2(e)': 1000.0, 'EX_cl(e)': 1000.0,
    'EX_zn2(e)': 1000.0, 'EX_so4(e)': 1000.0, 'EX_fe3(e)': 1000.0,
    'EX_h2o(e)': 1000.0, 'EX_k(e)': 1000.0, 'EX_nh4(e)': 1000.0,
    'EX_ca2(e)': 1000.0, 'EX_na1(e)': 1000.0, 'EX_fe2(e)': 1000.0
}

def compute_stationary_distribution(P, tol=1e-12):
    """
    Compute the stationary distribution π of the transition matrix P.
    π satisfies: P^T π = π, sum(π) = 1
    """
    n = P.shape[0]
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1)) 
    print(eigvecs)
    pi = np.real(eigvecs[:, idx])
    pi = pi / np.sum(pi)
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)  # Renormalize to make sure sum = 1
    return pi

def chung_laplacian_spectrum_directed(G):
    """
    Construct Chung's symmetric Laplacian for a directed graph G.
    Return the sorted real eigenvalues (spectrum).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Build weighted adjacency matrix A
    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        A[i, j] = data.get("weight", 1.0)
    
    # Build transition matrix P (row-normalized)
    row_sums = A.sum(axis=1, keepdims=True)
    P = np.divide(A, row_sums, where=row_sums != 0)
    
    # Compute stationary distribution π
    pi = compute_stationary_distribution(P)
    
    # Φ = diag(pi)
    sqrt_Phi = np.diag(np.sqrt(pi))
    inv_sqrt_Phi = np.diag(1 / np.sqrt(pi))
    
    # Chung symmetric Laplacian
    sym_P = 0.5 * (sqrt_Phi @ P @ inv_sqrt_Phi + inv_sqrt_Phi @ P.T @ sqrt_Phi)
    L = np.eye(n) - sym_P

    # Compute sorted eigenvalues
    spectrum = eigvalsh(L)  # Returns sorted real eigenvalues
    return spectrum



def get_inner_graph(G_coarse, inner_nodes):
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(inner_nodes)

    for u in inner_nodes:
        for v in inner_nodes:
            if u != v and G_coarse.has_edge(u, v):
                edge_data = G_coarse.get_edge_data(u, v)
                if edge_data is not None:
                    subgraph.add_edge(u, v, **edge_data)

    return subgraph

def construct_mass_flow_graph(S, met_flux, reaction_list):
    """
    Constructs a mass flow graph from the stoichiometric matrix S.
    """
    
    v = np.zeros(S.shape[1])

    uni = {}
    for met in met_flux:
            if met[1] == 0:
                continue
            if '_reverse' in met[0]:
                uni[met[0].replace('_reverse','')] = met[1] # Negative already
            elif '_forward' in met[0]:
                uni[met[0].replace('_forward','')] = met[1]
            else:
                uni[met[0]] = met[1]
    fixed_uni = [(key, value) for key, value in uni.items()]
    

    for key, value in fixed_uni:
        if key in reaction_list:
            v[reaction_list.index(key)] = value


    # for key, value in met_flux:
    #     if key in reaction_list:
    #         v[reaction_list.index(key)] = value
    
    v_forward = np.maximum(v, 0)
    v_backward = np.maximum(-v, 0)
    v_2m = np.concatenate([v_forward, v_backward])  # shape (2m,)

    # Step 2: Build S_2m^+ and S_2m^-
    S_plus = np.maximum(S, 0)
    S_minus = np.maximum(-S, 0)

    S2m_plus = np.concatenate([S_plus, S_minus], axis=1)   # shape (n, 2m)
    S2m_minus = np.concatenate([S_minus, S_plus], axis=1)  # flip roles for backward

    # print(S2m_plus.shape)
    # print(v_2m.shape)

    j_v = S2m_plus @ v_2m

    diag_v2m = np.diag(v_2m)
    diag_jv_inv = np.diag(np.where(j_v != 0, 1.0 / j_v, 0))

    M = (S2m_plus @ diag_v2m).T @ diag_jv_inv @ (S2m_minus @ diag_v2m)

    M[np.abs(M) < 1e-5] = 0

    return M

def construct_networkx_graph(M, reaction_list):
    G_mfg = nx.DiGraph()
    rxn_nodes = [f"{name}_fwd" for name in reaction_list] + [f"{name}_bwd" for name in reaction_list]

    for i in range(len(rxn_nodes)):
        for j in range(len(rxn_nodes)):
            if M[i, j] > 1e-5:  # threshold to show
                G_mfg.add_edge(rxn_nodes[i], rxn_nodes[j], weight=round(M[i, j], 3))
    # components = list(nx.weakly_connected_components(G_mfg))
    # largest_wcc = max(components, key=len)
    # G_largest = G_mfg.subgraph(largest_wcc).copy()

    return G_mfg


def walk_single_out_path(G, start_node):
    path = [start_node]
    current = start_node

    while True:
        successors = list(G.successors(current))
        if len(successors) != 1:
            break
        next_node = successors[0]
        
        if 'bio' in next_node or len(path) > 3:
            break
        path.append(next_node)
        current = next_node

    return path

def clustered_nodes4graph(G, met_flux, cluster_nodes):
    except_nodes = sorted([rec+'_bwd' for rec, value in met_flux if ('EX_' in rec and value < -0.01)])
    except_nodes_final = []
    for node in except_nodes:
        try:
            path = walk_single_out_path(G, node)
        except:
            except_nodes_final.append([node])
            continue
        if len(path) <= 3:
            except_nodes_final.append(path)
        else:
            except_nodes_final.append(path[:3])

    nodes = []
    for cluster_node in cluster_nodes.values():
        temp = []
        for node in cluster_node:
            if any(node in subgroup for subgroup in except_nodes_final) or (node not in list(G.nodes())):
                continue
            temp.append(node)
        nodes.append(temp)
    nodes.extend(except_nodes_final)

    return nodes, len(cluster_nodes)


def merge_nodes(graph, merged_nodes):
    """
    Merge nodes in the graph into clusters and update the edges' weights.
    
    Args:
        graph (nx.DiGraph): The original directed graph.
        merged_nodes (list): A list of lists of nodes that form clusters.
        
    Returns:
        coarse_grained_graph (nx.DiGraph): The new coarse-grained graph with updated edges.
    """
    coarse_grained_graph = nx.DiGraph()
    
    # Add cluster nodes
    cluster_to_nodes = {}
    for i, cluster in enumerate(merged_nodes):
        cluster_node = str(i)  # Use a simple integer as the cluster name
        coarse_grained_graph.add_node(cluster_node)  # Add the cluster as a node
        cluster_to_nodes[cluster_node] = cluster  # Map the cluster node to original nodes
    
    # Add edges between clusters based on the original graph
    for i, clusterA in enumerate(merged_nodes):
        for j, clusterB in enumerate(merged_nodes):
            if i == j:
                continue  # Skip the self-loops between the same clusters

            # Calculate the weight of the edge from clusterA to clusterB
            total_weight = 0
            edge_count = 0
            for nodeA in clusterA:
                for nodeB in clusterB:
                    if graph.has_edge(nodeA, nodeB):
                        total_weight += graph[nodeA][nodeB].get('weight', 1)
                        edge_count += 1
            
            # Standardize the weight by dividing by the number of nodes in clusterB
            if edge_count > 0:
                avg_weight = total_weight / edge_count
            else:
                avg_weight = 0
            
            if avg_weight > 0:
                # Add the edge with the calculated weight
                coarse_grained_graph.add_edge(str(i), str(j), weight=avg_weight)
    
    return coarse_grained_graph

def save_coarse_grained_graph(G, model, rst_index, nodes, filename):
    """
    Save the coarse-grained graph to a file.
    
    Args:
        graph (nx.DiGraph): The coarse-grained graph.
        filename (str): The name of the file to save the graph.
    """
    from collections import Counter
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    import textwrap

    fig, ax = plt.subplots(figsize=(32, 16))

    inner_radius = 0.6  
    outer_radius = 1.0  


    inner_nodes = [n for n in G.nodes if int(n) < rst_index] 
    outer_nodes = [n for n in G.nodes if int(n) >= rst_index]  

    pos = {}
    pos.update(nx.circular_layout(inner_nodes, scale=inner_radius)) 
    pos.update(nx.circular_layout(outer_nodes, scale=outer_radius)) 

    edge_colors = ['gray' for _ in G.edges()]
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    # normalized_weights = [20*w/max(edge_weights) for w in edge_weights]
    normalized_weights = [w for w in edge_weights]

    nx.draw_networkx_edges(G, pos, 
                        width=normalized_weights,
                        edge_color=edge_colors,
                        arrows=True,
                        arrowsize=20)


    nx.draw_networkx_nodes(G, pos,
                        nodelist=inner_nodes,
                        node_color='pink',
                        node_size=1000)

    nx.draw_networkx_nodes(G, pos,
                        nodelist=outer_nodes,
                        node_color='skyblue',
                        node_size=1000)

    labels_with_weights = {node: f"{node}" if node in inner_nodes else node for node in G.nodes}

    nx.draw_networkx_labels(G, pos, labels=labels_with_weights, font_size=20)

    legend_elements = []

    for i, group in enumerate(nodes):
        if i >= rst_index:
            pathway_str = ", ".join(group)
            wrapped_pathway_str = textwrap.fill(pathway_str, width=100)
            label_str = f'{i}: {wrapped_pathway_str}'
        else:
            subsystem_counter = Counter()
            for node in group:
                rxn_id = node[:-4]
                subsystem = model.reactions.get_by_id(rxn_id).subsystem or 'Unknown'
                subsystem_counter[subsystem] += 1

            subsystem_desc = ", ".join(f"{k}({v})" for k, v in subsystem_counter.most_common(5))

            label_str = f'{i} - {len(group)} nodes\n{subsystem_desc}'

        wrapped_label_str = textwrap.fill(label_str, width=100)

        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                    label=wrapped_label_str,
                    markersize=10,
                    markerfacecolor='skyblue' if i >= rst_index else 'pink')
        )
    ax.legend(handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def graph_node_atp_generation(model, G, met_flux, cluster_nodes):
    graph_atp_generation = {}

    for node_id in range(20):
        temp = node_atp_generation(model, G, met_flux, node_id, cluster_nodes)
        graph_atp_generation[node_id] = temp
    
    return graph_atp_generation

def node_atp_generation(model, G, met_flux, node_id, cluster_nodes):
        
    grouped_rec = [rec for rec in list(G.nodes) if rec in cluster_nodes[node_id]]
    grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in grouped_rec or rec+'_bwd' in grouped_rec}

    node_atp_generation = 0
    for rec, flux in grouped_flux.items():
        atp_generation = calculate_atp_generation(model, rec, flux)
        node_atp_generation += atp_generation

    return node_atp_generation

def calculate_atp_generation(model, rec_id, rec_flux):
        rec_temp = model.reactions.get_by_id(rec_id)
        keys = list(rec_temp.metabolites.keys())
        values = list(rec_temp.metabolites.values())

        if rec_flux == 0.0:
            return 0

        flux_generate = [[keys[i].id, values[i]] for i in range(len(keys)) if (values[i] > 0)] # 被指向
        flux_consume = [[keys[i].id, values[i]] for i in range(len(keys)) if (values[i] < 0)] # 指出去

        if any(item[0] == 'atp[c]' for item in flux_generate):
            w = next((item[1] for item in flux_generate if item[0] == 'atp[c]'), None)
        elif any(item[0] == 'atp[c]' for item in flux_consume):
            w = next((item[1] for item in flux_consume if item[0] == 'atp[c]'), None)
        else:
            w = 0

        return w * rec_flux

def node_proteome_cost(G, met_flux, node_id, cluster_nodes):
        
    grouped_rec = [rec for rec in list(G.nodes) if rec in cluster_nodes[node_id]]
    grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in grouped_rec or rec+'_bwd' in grouped_rec}

    node_proteome_cost = 0
    for rec, flux in grouped_flux.items():
        proteome_cost = abs(flux)
        node_proteome_cost += proteome_cost

    return node_proteome_cost

def graph_node_proteome_cost(G, met_flux, cluster_nodes):
    graph_proteome_cost = {}

    for node_id in range(20):
        temp = node_proteome_cost(G, met_flux, node_id, cluster_nodes)
        graph_proteome_cost[node_id] = temp

    return graph_proteome_cost



def graph_edge_proteome_cost(G, met_flux, cluster_nodes):
    edge_proteome_cost_matrix = np.zeros((20, 20))

    for i, j in [(i, j) for i in range(20) for j in range(20) if i!=j]:

        source_nodes = cluster_nodes[i]
        source_grouped_rec = [rec for rec in list(G.nodes) if rec in source_nodes]
        source_grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in source_grouped_rec or rec+'_bwd' in source_grouped_rec}

        target_nodes = cluster_nodes[j]
        target_grouped_rec = [rec for rec in list(G.nodes) if rec in target_nodes]
        target_grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in target_grouped_rec or rec+'_bwd' in target_grouped_rec}

        edge_proteome_cost = 0
        for u in source_nodes:
            if u not in G.nodes:
                continue

            for v in G.successors(u):
                if v in target_nodes:
                    temp = G[u][v].get('weight', 0)
                    edge_proteome_cost += temp
        edge_proteome_cost_matrix[i, j] = edge_proteome_cost

        
    return edge_proteome_cost_matrix



def pearson_ci(x, y, confidence=0.95):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n < 4:
        raise ValueError("Pearsonr CL requires at least 4 pairs of valid data.")

    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("x or y is constant, cannot compute correlation.")

    r, p = stats.pearsonr(x, y)

    r_clip = np.clip(r, -0.999999999, 0.999999999)

    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha / 2)

    z = np.arctanh(r_clip)
    se = 1 / np.sqrt(n - 3)

    ci_low = np.tanh(z - z_critical * se)
    ci_high = np.tanh(z + z_critical * se)

    return {
        "r": r,
        "p": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n
    }


def spearman_ci(
    x,
    y,
    confidence=0.95,
    n_boot=10000,
    random_state=42
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n < 4:
        raise ValueError("Spearman CL requires at least 4 pairs of valid data.")

    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("x or y is constant, cannot compute correlation.")

    rho, p = stats.spearmanr(x, y)

    rng = np.random.default_rng(random_state)
    bootstrap_rhos = []

    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)

        x_boot = x[indices]
        y_boot = y[indices]

        # 某次重采样后可能全部变成相同值
        if np.std(x_boot) == 0 or np.std(y_boot) == 0:
            continue

        rho_boot, _ = stats.spearmanr(x_boot, y_boot)

        if np.isfinite(rho_boot):
            bootstrap_rhos.append(rho_boot)

    if len(bootstrap_rhos) == 0:
        raise ValueError("Bootstrap not successful: all resampled pairs resulted in constant values.")

    bootstrap_rhos = np.asarray(bootstrap_rhos)

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_rhos, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2))

    return {
        "rho": rho,
        "p": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n
    }

def get_carbon_number(formula):
    match = re.search(r"C(\d*)", formula)

    if not match:
        return 0

    return int(match.group(1)) if match.group(1) else 1

def carbon_input_flux(met_flux, model):
    carbon_flux = 0
    for rec, flux in met_flux:
        if flux < 0 and 'EX_' in rec and rec not in shared_medium.keys():
            metabolite = list(model.reactions.get_by_id(rec).metabolites.keys())[0]
            formula = metabolite.formula
            carbon_number = get_carbon_number(formula)
            carbon_flux += abs(flux) * carbon_number
    return carbon_flux


def set_panel_style(scale: float = 1.0):
    fs = 7 * scale  # base font size
    mpl.rcParams.update({
        # --- Font ---
        "font.family": "Arial",
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "legend.fontsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,

        # --- Embed fonts in PDF/PS ---
        "pdf.fonttype": 42,  # TrueType; embed/subset fonts
        "ps.fonttype": 42,

        # --- Line weights (pt) ---
        "lines.linewidth": 1.1 * scale,
        "axes.linewidth": 0.9 * scale,
        "xtick.major.width": 0.8 * scale,
        "ytick.major.width": 0.8 * scale,
        "xtick.minor.width": 0.6 * scale,
        "ytick.minor.width": 0.6 * scale,

        # Tick lengths (pt)
        "xtick.major.size": 3.0 * scale,
        "ytick.major.size": 3.0 * scale,
        "xtick.minor.size": 1.8 * scale,
        "ytick.minor.size": 1.8 * scale,

        # --- Export hygiene ---
        "savefig.transparent": False,
    })