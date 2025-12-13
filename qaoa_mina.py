# qaoa_mina_final.py
# Archivo final QAOA (warm-start válido) para el reto TSP/VRP.
# Ejecutar: python qaoa_mina_final.py
# Requiere: pennylane, networkx, matplotlib, numpy

import numpy as np
import pennylane as qml
import networkx as nx
import matplotlib.pyplot as plt
import itertools

#Matriz de distancias
w = np.array([
    [0.0, 5.0, 8.0, 6.0],
    [5.0, 0.0, 3.0, 7.0],
    [8.0, 3.0, 0.0, 4.0],
    [6.0, 7.0, 4.0, 0.0]
], dtype=float)

def fuel_loaded(dist_ij, slope_ij):
    return 1.104 + 4.810 * slope_ij + 0.000024 * dist_ij

def fuel_unloaded(dist_ij, slope_ij):
    return 0.496 + 2.072 * slope_ij + 0.000014 * dist_ij

#parámetros a usar
A = 1200.0
B = 1200.0
C = 1.0
alpha_dijkstra = 0.0
p_layers = 3
opt_steps = 200
use_xy_mixer = True
warm_start = True


def idx(i, p, n=None):
    """Mapeo (i,p) -> índice de qubit"""
    if n is None:
        n = w.shape[0]
    positions = n - 1
    return (p - 1) * n + i

def compute_dijkstra_norm(w, depot=0):
    n = w.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cij = w[i, j]
            if np.isfinite(cij) and cij > 0:
                G.add_edge(i, j, weight=float(cij))
    try:
        length = nx.single_source_dijkstra_path_length(G, depot, weight='weight')
    except Exception:
        length = {i: np.inf for i in range(n)}
    dists = np.array([length.get(i, np.inf) for i in range(n)])
    finite_mask = np.isfinite(dists)
    if not finite_mask.any():
        return np.zeros(n)
    maxd = np.max(dists[finite_mask])
    dists[~finite_mask] = maxd * 10.0
    dmin, dmax = np.min(dists), np.max(dists)
    if dmax - dmin < 1e-9:
        return np.zeros(n)
    return (dists - dmin) / (dmax - dmin)

def build_cost_hamiltonian(w, A=10.0, B=10.0, C=1.0, alpha_dijkstra=0.0):
    n = w.shape[0]
    positions = n - 1
    num_qubits = n * positions

    const = 0.0
    lin = np.zeros(num_qubits)
    quad = {}

    def add_quad(q1, q2, delta):
        if q1 == q2:
            lin[q1] += delta
        else:
            key = tuple(sorted((q1, q2)))
            quad[key] = quad.get(key, 0.0) + delta

    for i in range(n):
        for j in range(n):
            cij = w[i, j]
            if not np.isfinite(cij) or abs(cij) < 1e-12:
                continue
            for p in range(1, positions):
                add_quad(idx(i, p), idx(j, p + 1), cij)

    for i in range(n):
        lin[idx(i, 1)] += w[0, i]
        lin[idx(i, positions)] += w[i, 0]

    for p in range(1, positions + 1):
        qs = [idx(i, p) for i in range(n)]
        const += A * 1.0
        for q in qs:
            lin[q] += -A
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * A)

    for i in range(1, n):
        qs = [idx(i, p) for p in range(1, positions + 1)]
        const += B * 1.0
        for q in qs:
            lin[q] += -B
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * B)

    ##idea de dijkstra porque no lo suelto jeje
    if alpha_dijkstra is not None and alpha_dijkstra > 0.0:
        d_norm = compute_dijkstra_norm(w, depot=0)
        for i in range(n):
            penalty_i = alpha_dijkstra * float(d_norm[i])
            for p in range(1, positions + 1):
                lin[idx(i, p)] += penalty_i

    const_z = const
    lin_z = np.zeros(num_qubits)
    quad_z = {}
    for q, a in enumerate(lin):
        if abs(a) < 1e-12:
            continue
        const_z += 0.5 * a
        lin_z[q] += -0.5 * a
    for (q1, q2), b in quad.items():
        const_z += C * 0.25 * b
        lin_z[q1] += -C * 0.25 * b
        lin_z[q2] += -C * 0.25 * b
        key = (q1, q2)
        quad_z[key] = quad_z.get(key, 0.0) + C * 0.25 * b

    coeffs = []
    ops = []
    if abs(const_z) > 1e-12:
        coeffs.append(const_z)
        ops.append(qml.Identity(0))
    for q in range(num_qubits):
        c = lin_z[q]
        if abs(c) > 1e-12:
            coeffs.append(c)
            ops.append(qml.PauliZ(q))
    for (q1, q2), c in quad_z.items():
        if abs(c) > 1e-12:
            coeffs.append(c)
            ops.append(qml.PauliZ(q1) @ qml.PauliZ(q2))

    H_cost = qml.Hamiltonian(coeffs, ops)
    return H_cost, lin, quad, const

def build_xy_mixer(num_cities):
    n = num_cities
    positions = n - 1
    num_qubits = n * positions
    terms = []
    coeffs = []
    for p in range(1, positions + 1):
        block = [idx(i, p, n) for i in range(n)]
        for a in range(len(block)):
            for b in range(a + 1, len(block)):
                q1 = block[a]
                q2 = block[b]
                terms.append(qml.PauliX(q1) @ qml.PauliX(q2))
                coeffs.append(1.0)
                terms.append(qml.PauliY(q1) @ qml.PauliY(q2))
                coeffs.append(1.0)
    return qml.Hamiltonian(coeffs, terms)

def build_x_mixer(num_cities):
    num_qubits = num_cities * (num_cities - 1)
    terms = []
    coeffs = []
    for q in range(num_qubits):
        terms.append(qml.PauliX(q))
        coeffs.append(1.0)
    return qml.Hamiltonian(coeffs, terms)

def decode_route(bitstring, num_cities):
    n = num_cities
    positions = n - 1
    route = [0]
    visited = set()
    valid = True
    reason = None
    for p in range(1, positions + 1):
        segment = bitstring[(p - 1) * n : p * n]
        ones = [i for i, ch in enumerate(segment) if ch == "1"]
        if len(ones) != 1:
            valid = False
            reason = f"posición {p} tiene {len(ones)} unos"
            route.append(None)
            continue
        city = ones[0]
        if city == 0:
            valid = False
            reason = f"posición {p} eligió depósito (0)"
            route.append(None)
            continue
        if city in visited:
            valid = False
            reason = f"ciudad {city} repetida"
        visited.add(city)
        route.append(city)
    if visited != set(range(1, n)):
        valid = False
        if reason is None:
            missing = sorted(set(range(1, n)) - visited)
            reason = f"faltan ciudades {missing}"
    route.append(0)
    return route, valid, reason

def classical_cost_of_route(route, wmat):
    if any(r is None for r in route):
        return float('inf')
    c = 0.0
    for i in range(len(route) - 1):
        c += wmat[route[i], route[i+1]]
    return c

def best_classical_route(wmat):
    n = wmat.shape[0]
    best_cost = None
    best_route = None
    for perm in itertools.permutations(range(1, n)):
        route = [0] + list(perm) + [0]
        c = classical_cost_of_route(route, wmat)
        if best_cost is None or c < best_cost - 1e-9:
            best_cost = c
            best_route = route
    return best_route, best_cost

best_route_classic, best_cost_classic = best_classical_route(w)
print("Mejor ruta clásica (para warm-start):", best_route_classic, "costo:", best_cost_classic)

def route_to_bitstring(route, n):
    positions = n - 1
    bits = ['0'] * (n * positions)
    for p in range(1, positions + 1):
        city = route[p]
        q = idx(city, p, n)
        bits[q] = '1'
    return ''.join(bits)

bitstring_warm = route_to_bitstring(best_route_classic, w.shape[0])
print("Bitstring warm-start (base):", bitstring_warm)

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    n = w.shape[0]
    positions = n - 1
    num_qubits = n * positions

    H_cost, lin_poly, quad_poly, const_poly = build_cost_hamiltonian(
        w, A=A, B=B, C=C, alpha_dijkstra=alpha_dijkstra
    )

    H_mixer = build_xy_mixer(n) if use_xy_mixer else build_x_mixer(n)

    print("H_cost terms:", len(H_cost.ops))
    print("H_mixer terms:", len(H_mixer.ops))
    print("num_qubits:", num_qubits)

    dev = qml.device("default.qubit", wires=num_qubits)

    def qaoa_layer(gamma, beta):
        qml.ApproxTimeEvolution(H_cost, gamma, 1)
        qml.ApproxTimeEvolution(H_mixer, beta, 1)

    @qml.qnode(dev)
    def circuit(params, warm=False):
        if warm:
            for q_idx, bit in enumerate(bitstring_warm):
                if bit == '1':
                    qml.PauliX(wires=q_idx)
        else:
            for q in range(num_qubits):
                qml.Hadamard(q)
        for layer in range(p_layers):
            qaoa_layer(params[0][layer], params[1][layer])
        return qml.expval(H_cost)

    @qml.qnode(dev)
    def probs_qnode(params, warm=False):
        if warm:
            for q_idx, bit in enumerate(bitstring_warm):
                if bit == '1':
                    qml.PauliX(wires=q_idx)
        else:
            for q in range(num_qubits):
                qml.Hadamard(q)
        for layer in range(p_layers):
            qaoa_layer(params[0][layer], params[1][layer])
        return qml.probs(wires=range(num_qubits))

    rng = np.random.default_rng(42)
    init_params = rng.uniform(0, 2 * np.pi, (2, p_layers))
    params = qml.numpy.array(init_params, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=0.05)

    print(f"\nEjecutando QAOA p={p_layers} (warm_start={warm_start})...\n")
    for it in range(opt_steps):
        params = opt.step(lambda p: circuit(p, warm=warm_start), params)
        if (it + 1) % 10 == 0 or it == 0:
            energy = circuit(params, warm=warm_start)
            print(f"Iter {it:4d} | Energy = {energy:.6f}")

    print("\nParámetros óptimos:\n", params)

    probs = np.array(probs_qnode(params, warm=warm_start))

    top_k = 20
    sorted_indices = np.argsort(probs)[::-1][:top_k]
    print("\nTop-{} estados decodificados:".format(top_k))

    candidate_solutions = []
    for idx_i in sorted_indices:
        bs = format(idx_i, f"0{num_qubits}b")
        route, valid, reason = decode_route(bs, n)
        prob = probs[idx_i]
        E_class = classical_cost_of_route(route, w) if valid else classical_cost_of_route(route, w)
        print(f"{bs} -> {route}  (p={prob:.6f})  valid={valid}  reason={reason}  E_class={E_class:.6f}")
        candidate_solutions.append((bs, route, valid, prob, E_class, reason))

    valids = [c for c in candidate_solutions if c[2]]
    if valids:
        best = min(valids, key=lambda x: x[4])
        print("\nMejor solución válida encontrada entre top-K:")
        print(f"{best[0]} -> {best[1]}   prob={best[3]:.6f}  E_class={best[4]:.6f}")
        best_route = best[1]
        best_valid = True
    else:
        print("\nNo se encontró solución válida entre top-K. Tomando la más probable (aunque inválida).")
        best = candidate_solutions[0]
        best_route = best[1]
        best_valid = False

    clean_route = [r for r in best_route if r is not None]
    if best_valid and len(clean_route) >= 2:
        G = nx.DiGraph()
        nodes = list(range(n))
        G.add_nodes_from(nodes)
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(i, j, weight=float(w[i, j]) if np.isfinite(w[i, j]) else 1e6)
        path_edges = [(clean_route[i], clean_route[i + 1]) for i in range(len(clean_route) - 1)]
        pos_layout = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(7, 6))
        nx.draw_networkx_nodes(G, pos_layout, node_color='lightgray', node_size=700)
        nx.draw_networkx_labels(G, pos_layout, font_size=12)
        nx.draw_networkx_edges(G, pos_layout, edge_color='lightgray')
        nx.draw_networkx_edges(G, pos_layout, edgelist=path_edges, edge_color='red', width=3)
        plt.title("Ruta (decodificada) según QAOA (mejor válida)")
        plt.axis('off')
        plt.show()
    else:
        print("No se pudo generar una ruta graficable (no hay solución válida entre top-K).")

    # Gráfica de probabilidades top-K
    top_labels = [format(i, f"0{num_qubits}b") for i in sorted_indices]
    top_probs = probs[sorted_indices]
    plt.figure(figsize=(10, 4))
    plt.bar(top_labels, top_probs)
    plt.xticks(rotation=90)
    plt.title("Top-{} estados más probables (QAOA)".format(top_k))
    plt.tight_layout()
    plt.show()

    # Gráfica completa (opcional, se puede comentar si 2^num_qubits grande)
    labels = [format(i, f"0{num_qubits}b") for i in range(len(probs))]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, probs)
    plt.xticks(rotation=90)
    plt.title("Distribución de probabilidades QAOA (completa)")
    plt.tight_layout()
    plt.show()
