import numpy as np
import pennylane as qml
import networkx as nx
import matplotlib.pyplot as plt

##Matriz de costos de ejemplo
w = np.array([
    [0.0, 5.0, 8.0, 6.0], 
    [5.0, 0.0, 3.0, 7.0],
    [8.0, 3.0, 0.0, 4.0],
    [6.0, 7.0, 4.0, 0.0]
])
A = 10.0
B = 10.0
C = 1.0


def idx(i, p, n=None):
    
    if n is None:
        n = w.shape[0]
    positions = n - 1
    return (p-1) * n + i

## trato de copiar el comportamiento de dijkstra
def compute_dijkstra_norm(w, depot=0):
    n = w.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cij = w[i,j]
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
    norm = (dists - dmin) / (dmax - dmin)
    return norm

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
            if abs(cij) < 1e-12:
                continue
            for p in range(1, positions):
                q_ip = idx(i, p, n)
                q_jp1 = idx(j, p + 1, n)
                add_quad(q_ip, q_jp1, cij)

    for i in range(n):
        q_i1 = idx(i, 1, n)
        lin[q_i1] += w[0, i]

    for i in range(n):
        q_i_last = idx(i, positions, n)
        lin[q_i_last] += w[i, 0]

    for p in range(1, positions + 1):
        qs = [idx(i, p, n) for i in range(n)]
        const += A * 1.0
        for q in qs:
            lin[q] += -A
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * A)

    for i in range(1, n): 
        qs = [idx(i, p, n) for p in range(1, positions + 1)]
        const += B * 1.0
        for q in qs:
            lin[q] += -B
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * B)

    if alpha_dijkstra is not None and alpha_dijkstra > 0.0:
        d_norm = compute_dijkstra_norm(w, depot=0)  # normalizado 0..1
        for i in range(n):
            penalty_i = alpha_dijkstra * d_norm[i]
            for p in range(1, positions + 1):
                q = idx(i, p, n)
                lin[q] += penalty_i

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
    return H_cost

##Mixer
def build_xy_mixer(num_cities):
    num_qubits = num_cities * (num_cities - 1)
    terms = []
    coeffs = []
    clients = list(range(1, num_cities))  
    positions = num_cities - 1
    for pos in range(1, positions + 1):
        block = [idx(c, pos, num_cities) for c in clients]
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                q1 = block[i]
                q2 = block[j]
                terms.append(qml.PauliX(q1) @ qml.PauliX(q2))
                coeffs.append(1.0)
                terms.append(qml.PauliY(q1) @ qml.PauliY(q2))
                coeffs.append(1.0)
    H_mixer = qml.Hamiltonian(coeffs, terms)
    return H_mixer

if __name__ == "__main__":
    num_cities = w.shape[0]
    H_cost = build_cost_hamiltonian(w, A=A, B=B, C=C, alpha_dijkstra=2.0)
    print("H_cost terms:", len(H_cost.coeffs))
    H_mixer = build_xy_mixer(num_cities)
    print("H_mixer terms:", len(H_mixer.coeffs))
    print("H_cost coeffs (first 10):", H_cost.coeffs[:10])
    print("H_mixer coeffs (first 10):", H_mixer.coeffs[:10])




##Pruebita    
    print("\nEjecutando QAOA p=1...\n")

    import pennylane as qml
    from pennylane import numpy as pnp

    num_qubits = (num_cities) * (num_cities - 1)
    dev = qml.device("default.qubit", wires=num_qubits)

    def qaoa_layer(gamma, beta):
        qml.expval  
        qml.ApproxTimeEvolution(H_cost, gamma, 1)
        qml.ApproxTimeEvolution(H_mixer, beta, 1)

    @qml.qnode(dev)
    def circuit(params):
        for q in range(num_qubits):
            qml.Hadamard(q)

        gamma, beta = params
        qaoa_layer(gamma, beta)

        return qml.expval(H_cost)

    opt = qml.AdamOptimizer(stepsize=0.04)
    params = pnp.array([0.1, 0.1], requires_grad=True)

    for it in range(40):
        params = opt.step(circuit, params)
        if it % 10 == 0:
            print(f"Iter {it:3d} | Energy = {circuit(params):.4f}")

    print("\nParámetros óptimos:", params)
    

    @qml.qnode(dev)
    def probs(params):
        for q in range(num_qubits):
            qml.Hadamard(q)
        gamma, beta = params
        qaoa_layer(gamma, beta)
        return qml.probs(range(num_qubits))

    p = probs(params)

    ## ---------- Gráfica limpia: Top 10 probabilidades ----------
    top_k = 10
    sorted_indices = np.argsort(p)[::-1][:top_k]
    top_probs = p[sorted_indices]
    top_labels = [format(i, f"0{num_qubits}b") for i in sorted_indices]

    plt.figure(figsize=(10,5))
    plt.bar(top_labels, top_probs)
    plt.xticks(rotation=90)
    plt.title("Top 10 estados más probables (QAOA)")
    plt.xlabel("Bitstring")
    plt.ylabel("Probabilidad")
    plt.tight_layout()
    plt.show()

    import numpy as np
    max_index = np.argmax(p)
    bitstring = format(max_index, f"0{num_qubits}b")
    print("\nSolución más probable:", bitstring)
    print("Probabilidad:", p[max_index])
    
    def decode_route(bitstring, num_cities):
        n = num_cities
        positions = n - 1
        route = [0]  

        for p in range(1, positions + 1):
            segment = bitstring[(p - 1) * n : p * n]
            if "1" in segment:
                i = segment.index("1")
                if i != 0:   
                    route.append(i)
            else:
                route.append(None)

        route.append(0)  
        return route

    route = decode_route(bitstring, num_cities)
    print("\nRuta decodificada:", route)
    
        ## ---------- Graficar la ruta como grafo ----------
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()

    # Agregar nodos
    nodes = list(range(num_cities))
    G.add_nodes_from(nodes)

    # Agregar todas las aristas (con pesos)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                G.add_edge(i, j, weight=w[i, j])

    # Extraer la ruta limpia sin "None"
    clean_route = [node for node in route if node is not None]

    # Crear lista de aristas de la ruta encontrada
    path_edges = [(clean_route[i], clean_route[i+1]) for i in range(len(clean_route)-1)]

    pos = nx.spring_layout(G, seed=42)  # para que quede bonito y estable

    plt.figure(figsize=(7, 6))

    # Dibujar todos los nodos y aristas en gris
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')

    # Dibujar la ruta óptima en ROJO
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='red', width=3)

    plt.title("Ruta óptima según QAOA")
    plt.axis('off')
    plt.show()




    labels = [format(i, f"0{num_qubits}b") for i in range(len(p))]

    plt.figure(figsize=(12,6))
    plt.bar(labels, p)
    plt.xticks(rotation=90)
    plt.title("Distribución de probabilidades QAOA")
    plt.xlabel("Bitstring")
    plt.ylabel("Probabilidad")
    plt.tight_layout()
    plt.show()
