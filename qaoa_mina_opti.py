import numpy as np
import pennylane as qml
import networkx as nx
import matplotlib.pyplot as plt
import itertools

# Depósito + 2 Frentes + 1 Botadero
# Distancias en km según operación real
w = np.array([
    [0.0, 2.5, 4.1, 6.5],  # 0: Depósito
    [2.5, 0.0, 4.2, 5.8],  # 1: Frente_Norte
    [4.1, 4.2, 0.0, 7.3],  # 2: Frente_Centro
    [6.5, 5.8, 7.3, 0.0],  # 3: Botadero_Principal
], dtype=float)

LOCATION_NAMES = {
    0: "Deposito",
    1: "Frente_Norte", 
    2: "Frente_Centro",
    3: "Botadero_Principal"
}

A = 75.0   # Penalización unicidad espacial (distancias reales)
B = 75.0   # Penalización unicidad temporal
C = 1.0    # Factor de distancia
alpha_dijkstra = 0.0
p_layers = 5
opt_steps = 300
use_xy_mixer = True
warm_start = False


print(f"  Ubicaciones: {w.shape[0]}")
print(f"  Qubits: {w.shape[0] * (w.shape[0]-1)}")
print(f"  Estados cuánticos: 2^{w.shape[0] * (w.shape[0]-1)} = {2**(w.shape[0] * (w.shape[0]-1)):,}")
print(f"  Penalizaciones: A={A}, B={B}, C={C}")
print(f"  Layers QAOA: {p_layers}")
print(f"  Pasos optimización: {opt_steps}")
print(f"  Mixer: {'XY (restricciones)' if use_xy_mixer else 'X (estándar)'}")
print(f"  Warm start: {warm_start}")
print("="*70)
print()

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

    # Función objetivo: distancia
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

    # Restricción: unicidad espacial
    for p in range(1, positions + 1):
        qs = [idx(i, p) for i in range(n)]
        const += A * 1.0
        for q in qs:
            lin[q] += -A
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * A)

    # Restricción: unicidad temporal
    for i in range(1, n):
        qs = [idx(i, p) for p in range(1, positions + 1)]
        const += B * 1.0
        for q in qs:
            lin[q] += -B
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * B)

    if alpha_dijkstra is not None and alpha_dijkstra > 0.0:
        d_norm = compute_dijkstra_norm(w, depot=0)
        for i in range(n):
            penalty_i = alpha_dijkstra * float(d_norm[i])
            for p in range(1, positions + 1):
                lin[idx(i, p)] += penalty_i

    # Conversión QUBO a Pauli
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
    """Mixer que preserva restricciones (X_i X_j + Y_i Y_j)"""
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
    """Mixer estándar (solo X gates)"""
    num_qubits = num_cities * (num_cities - 1)
    terms = []
    coeffs = []
    for q in range(num_qubits):
        terms.append(qml.PauliX(q))
        coeffs.append(1.0)
    return qml.Hamiltonian(coeffs, terms)

def decode_route(bitstring, num_cities):
    """Decodifica bitstring a ruta"""
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
    """Calcula costo de una ruta en la matriz de distancias"""
    if any(r is None for r in route):
        return float('inf')
    c = 0.0
    for i in range(len(route) - 1):
        c += wmat[route[i], route[i+1]]
    return c

def best_classical_route(wmat):
    """Encuentra la mejor ruta por fuerza bruta"""
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

# Calcular óptimo clásico
best_route_classic, best_cost_classic = best_classical_route(w)
print("REFERENCIA CLÁSICA (Fuerza Bruta):")
route_names = [LOCATION_NAMES[r] for r in best_route_classic]
print(f"   Ruta óptima: {' → '.join(route_names)}")
print(f"   Costo óptimo: {best_cost_classic:.2f} km")
print()

def route_to_bitstring(route, n):
    """Convierte ruta a bitstring para warm-start"""
    positions = n - 1
    bits = ['0'] * (n * positions)
    for p in range(1, positions + 1):
        city = route[p]
        q = idx(city, p, n)
        bits[q] = '1'
    return ''.join(bits)

bitstring_warm = route_to_bitstring(best_route_classic, w.shape[0])
print(f"Bitstring warm-start: {bitstring_warm}")
print()

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    n = w.shape[0]
    positions = n - 1
    num_qubits = n * positions

    H_cost, lin_poly, quad_poly, const_poly = build_cost_hamiltonian(
        w, A=A, B=B, C=C, alpha_dijkstra=alpha_dijkstra
    )

    H_mixer = build_xy_mixer(n) if use_xy_mixer else build_x_mixer(n)

    print(f"HAMILTONIANO:")
    print(f"H_cost terms: {len(H_cost.ops)}")
    print(f"H_mixer terms: {len(H_mixer.ops)}")
    print()

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

    print(f"{'='*70}")
    print(f" EJECUTANDO QAOA (p={p_layers}, warm_start={warm_start})")
    print(f"{'='*70}\n")
    
    for it in range(opt_steps):
        params = opt.step(lambda p: circuit(p, warm=warm_start), params)
        if (it + 1) % 10 == 0 or it == 0:
            energy = circuit(params, warm=warm_start)
            print(f"Iter {it:4d} | Energy = {energy:.6f}")

    print(f"\n Optimización completada")
    print(f"   Parámetros óptimos: {params.shape}")
    print()

    probs = np.array(probs_qnode(params, warm=warm_start))

    top_k = 20
    sorted_indices = np.argsort(probs)[::-1][:top_k]
    
    print(f"{'='*70}")
    print(f" TOP-{top_k} ESTADOS DECODIFICADOS")
    print(f"{'='*70}\n")

    candidate_solutions = []
    valid_count = 0
    
    for rank, idx_i in enumerate(sorted_indices, 1):
        bs = format(idx_i, f"0{num_qubits}b")
        route, valid, reason = decode_route(bs, n)
        prob = probs[idx_i]
        E_class = classical_cost_of_route(route, w)
        
        if valid:
            valid_count += 1
        
        # Traducir ruta a nombres
        route_names = []
        for r in route:
            if r is not None:
                route_names.append(LOCATION_NAMES[r])
            else:
                route_names.append("???")
        
        status = "✓" if valid else "✗"
        print(f"{rank:2d}. {status} [{bs}]")
        print(f"    Ruta: {' → '.join(route_names)}")
        print(f"    Prob={prob:.4f} | Costo={E_class:.2f} km | Valid={valid}")
        if not valid:
            print(f"    Razón: {reason}")
        print()
        
        candidate_solutions.append((bs, route, valid, prob, E_class, reason))

    valids = [c for c in candidate_solutions if c[2]]
    
    print(f"\n{'='*70}")
    print(f" RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"  Soluciones válidas: {len(valids)}/{top_k} ({len(valids)/top_k*100:.1f}%)")
    print()
    
    if valids:
        best = min(valids, key=lambda x: x[4])
        route_names = [LOCATION_NAMES[r] for r in best[1] if r is not None]
        gap = ((best[4] - best_cost_classic) / best_cost_classic) * 100
        
        print(f"MEJOR SOLUCIÓN VÁLIDA (QAOA):")
        print(f"Ruta: {' → '.join(route_names)}")
        print(f"Costo QAOA: {best[4]:.2f} km")
        print(f"Costo óptimo: {best_cost_classic:.2f} km")
        print(f"GAP: {gap:.2f}%")
        print(f"Probabilidad: {best[3]:.4f}")
        print()
        
        # Análisis según paper de literatura
        if gap <= 5:
            print(f"EXCELENTE: Gap <5% (mejor que D-Wave reportado)")
        elif gap <= 15:
            print(f"BUENO: Gap en rango esperado 5-15% según literatura")
        elif gap <= 25:
            print(f"ACEPTABLE: Gap 15-25%, considerar aumentar layers")
        else:
            print(f"ALTO: Gap >{gap:.1f}%, ajustar A, B o layers")
        
        best_route = best[1]
        best_valid = True
    else:
        print(f"NO HAY SOLUCIONES VÁLIDAS")
        print(f"   Acción: Aumentar A, B a {A*2:.0f} o más")
        best = candidate_solutions[0]
        best_route = best[1]
        best_valid = False

    # Visualización
    clean_route = [r for r in best_route if r is not None]
    if best_valid and len(clean_route) >= 2:
        G = nx.DiGraph()
        nodes = list(range(n))
        G.add_nodes_from(nodes)
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(i, j, weight=float(w[i, j]))
        
        path_edges = [(clean_route[i], clean_route[i + 1]) for i in range(len(clean_route) - 1)]
        pos_layout = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos_layout, node_color='lightblue', node_size=1500)
        
        labels = {i: f"{i}\n{LOCATION_NAMES[i]}" for i in range(n)}
        nx.draw_networkx_labels(G, pos_layout, labels, font_size=10, font_weight='bold')
        
        nx.draw_networkx_edges(G, pos_layout, edge_color='lightgray', alpha=0.3)
        nx.draw_networkx_edges(G, pos_layout, edgelist=path_edges, 
                              edge_color='red', width=4, arrows=True, 
                              arrowsize=25, arrowstyle='->', connectionstyle='arc3,rad=0.1')
        
        title = f"Ruta Óptima QAOA - AngloAmerican\nCosto: {best[4]:.2f} km | Gap: {gap:.2f}%"
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('ruta_qaoa_angloamerican.png', dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: ruta_qaoa_angloamerican.png\n")
        plt.show()
    else:
        print("No se generó visualización (sin solución válida)\n")

    # Gráfica de probabilidades
    top_labels = [f"{i}" for i in range(1, top_k+1)]
    top_probs = probs[sorted_indices]
    colors = ['green' if candidate_solutions[i][2] else 'red' for i in range(top_k)]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(top_labels, top_probs, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Probabilidad', fontsize=12)
    plt.title(f'Top-{top_k} Estados QAOA (Verde=Válido, Rojo=Inválido)\n'
              f'Válidos: {valid_count}/{top_k}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('probabilidades_qaoa.png', dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: probabilidades_qaoa.png\n")
    plt.show()
    print("="*70)
    print(" EJECUCIÓN COMPLETADA")
    print("="*70)




