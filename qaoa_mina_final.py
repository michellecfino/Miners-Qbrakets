#!/usr/bin/env python3
"""
QAOA Mining Route Optimizatio
Challenge 2: Miners Qbrakets Team

Combina:
- Datos de AngloAmerican Peru (validados)
- Datos de paper de Indonesia (con pendientes)
- Implementacion del optimizador de Unai
- Implementación de mixer XY de Andres
- Implementacion de mixer SWAP de Charlie
- Penalizaciones calibradas

"""

import numpy as np
import pennylane as qml
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import os

# ============================================================================
# SECCION DE CONFIGURACION
# ============================================================================

# Seleccionar dataset interactivamente
print(" QAOA - OPTIMIZACION DE RUTAS MINERAS")
print("\nDatasets disponibles:")
print("  1. Peru - AngloAmerican (datos validados)")
print("  2. Indonesia - Paper academico (con pendientes)")
print()

while True:
    choice = input("Seleccione dataset (1 o 2): ").strip()
    if choice == '1':
        DATASET = 'peru'
        break
    elif choice == '2':
        DATASET = 'indonesia'
        break
    else:
        print("Opcion invalida. Por favor ingrese 1 o 2.")

print(f"\nDataset seleccionado: {DATASET.upper()}")
print("=" * 70)

# Parametros QAOA
p_layers = 3          # Numero de capas QAOA (optimo para este problema)
opt_steps = 200       # Pasos de optimizacion clasica
warm_start = True     # CRITICO: Enfoque hibrido quantum-clasico

# ============================================================================
# SELECCION INTERACTIVA DEL MIXER
# ============================================================================

print("\nMixers disponibles:")
print("  1. X  (estándar, NO preserva restricciones)")
print("  2. XY (preserva one-hot por posición)")
print("  3. SWAP (preserva tours válidos completos)")
print()

while True:
    mixer_choice = input("Seleccione mixer (1, 2 o 3): ").strip()
    if mixer_choice == '1':
        MIXER_TYPE = 'x'
        break
    elif mixer_choice == '2':
        MIXER_TYPE = 'xy'
        break
    elif mixer_choice == '3':
        MIXER_TYPE = 'swap'
        break
    else:
        print("Opción inválida. Ingrese 1, 2 o 3.")

print(f"\nMixer seleccionado: {MIXER_TYPE.upper()}")
print("=" * 70)

# ============================================================================
# DEFINICIONES DEL CONJUNTO DE DATOS
# ============================================================================

if DATASET == 'peru':
    # AngloAmerican Peru - Datos validados de la industria
    # Matriz simetrica (distancias planas en km)
    w = np.array([
        [0.0, 2.5, 4.1, 6.5],  # 0: Deposito
        [2.5, 0.0, 4.2, 5.8],  # 1: Frente_Norte
        [4.1, 4.2, 0.0, 7.3],  # 2: Frente_Centro
        [6.5, 5.8, 7.3, 0.0]   # 3: Botadero_Principal
    ], dtype=float)
    
    LOCATION_NAMES = {
        0: "Deposito",
        1: "Frente_Norte",
        2: "Frente_Centro",
        3: "Botadero_Principal"
    }
    
    # Penalizaciones calibradas (10x max_distance = 7.3 km)
    A = 75.0  # Penalizacion de unicidad espacial
    B = 75.0  # Penalizacion de unicidad temporal
    C = 1.0   # Factor de costo de distancia
    
    print("\nDATASET: AngloAmerican Peru (Datos Validados)")
    print("-" * 70)

elif DATASET == 'indonesia':
    # Indonesia - Datos del paper (Saptarini et al.)
    # Matriz asimetrica (consumo de combustible con pendientes)
    # Ecuaciones de combustible:
    #   cargado: 1.104 + 4.810*pendiente + 0.000024*distancia
    #   vacio: 0.496 + 2.072*pendiente + 0.000014*distancia
    
    w = np.array([
        [0.0,  25.18, 8.963, 1.14 ],  # 0: Palas (punto de carga)
        [0.51,  0.0,  0.51, 25.0  ],  # 1: Botadero esteril
        [0.51, 15.56, 0.0,  0.51  ],  # 2: Stock
        [0.51, 25.2, 10.74, 0.0   ]   # 3: Planta procesamiento
    ], dtype=float)
    
    LOCATION_NAMES = {
        0: "Palas",
        1: "Botadero_Esteril",
        2: "Stock",
        3: "Planta_Procesamiento"
    }
    
    # Penalizaciones originales del paper (NO MODIFICAR para Indonesia)
    A = 1200.0
    B = 1200.0
    C = 1.0
    
    print("\nDATASET: Operacion Minera Indonesia (Datos de Paper)")
    print("-" * 70)
    print("Matriz incluye consumo de combustible ajustado por pendiente")
    print("Asimetrica debido a diferencias cuesta arriba/abajo")

else:
    raise ValueError(f"Dataset desconocido: {DATASET}")

# Imprimir configuracion
n = w.shape[0]
positions = n - 1
num_qubits = n * positions

print(f"  Ubicaciones: {n}")
print(f"  Qubits: {num_qubits}")
print(f"  Estados cuanticos: 2^{num_qubits} = {2**num_qubits:,}")
print(f"  Penalizaciones: A={A}, B={B}, C={C}")
print(f"  Capas QAOA: {p_layers}")
print(f"  Pasos optimizacion: {opt_steps}")
mixer_labels = {
    'x':   'X (estándar, viola restricciones)',
    'xy':  'XY (one-hot por posición)',
    'swap':'SWAP (tours válidos completos)'
}

print(f"  Mixer: {mixer_labels[MIXER_TYPE]}")
print(f"  Warm start: {warm_start}")
print("=" * 70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def idx(i, p, n=None):
    """Map (location i, position p) to qubit index"""
    if n is None:
        n = w.shape[0]
    positions = n - 1
    return (p - 1) * n + i

def decode_route(bitstring, num_cities):
    """Decode bitstring to route and validate constraints"""
    n = num_cities
    positions = n - 1
    route = [0]  # Start at depot/loading point
    visited = set()
    valid = True
    reason = None
    
    for p in range(1, positions + 1):
        segment = bitstring[(p - 1) * n : p * n]
        ones = [i for i, ch in enumerate(segment) if ch == "1"]
        
        # Check exactly one location per position
        if len(ones) != 1:
            valid = False
            reason = f"posición {p} tiene {len(ones)} unos"
            route.append(None)
            continue
        
        city = ones[0]
        
        # Check not returning to depot mid-route
        if city == 0:
            valid = False
            reason = f"posición {p} eligió depósito (0)"
            route.append(None)
            continue
        
        # Check no repeated cities
        if city in visited:
            valid = False
            reason = f"ciudad {city} repetida"
        
        visited.add(city)
        route.append(city)
    
    # Check all cities visited
    if visited != set(range(1, n)):
        valid = False
        if reason is None:
            missing = sorted(set(range(1, n)) - visited)
            reason = f"faltan ciudades {missing}"
    
    route.append(0)  # Return to depot
    return route, valid, reason

def classical_cost_of_route(route, wmat):
    """Calculate total cost of a route"""
    if any(r is None for r in route):
        return float('inf')
    cost = 0.0
    for i in range(len(route) - 1):
        cost += wmat[route[i], route[i + 1]]
    return cost

def best_classical_route(wmat):
    """Brute force search for optimal route"""
    n = wmat.shape[0]
    best_cost = None
    best_route = None
    
    for perm in itertools.permutations(range(1, n)):
        route = [0] + list(perm) + [0]
        cost = classical_cost_of_route(route, wmat)
        
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_route = route
    
    return best_route, best_cost

def route_to_bitstring(route, n):
    """Convert route to bitstring for warm start"""
    positions = n - 1
    bits = ['0'] * (n * positions)
    
    for p in range(1, positions + 1):
        city = route[p]
        q = idx(city, p, n)
        bits[q] = '1'
    
    return ''.join(bits)

# ============================================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================================

def build_cost_hamiltonian(w, A=10.0, B=10.0, C=1.0):
    """
    Build cost Hamiltonian for TSP/VRP
    
    H_cost = C * Σ d_ij * x_i,t * x_j,t+1  (distance costs)
           + A * Σ (1 - Σ x_i,t)^2         (one city per position)
           + B * Σ (1 - Σ x_i,t)^2         (visit each city once)
    
    Returns Pauli-Z form for PennyLane
    """
    n = w.shape[0]
    positions = n - 1
    num_qubits = n * positions
    
    const = 0.0
    lin = np.zeros(num_qubits)
    quad = {}
    
    def add_quad(q1, q2, delta):
        """Helper to accumulate quadratic terms"""
        if q1 == q2:
            lin[q1] += delta
        else:
            key = tuple(sorted((q1, q2)))
            quad[key] = quad.get(key, 0.0) + delta
    
    # Distance costs (inter-position arcs)
    for i in range(n):
        for j in range(n):
            cij = w[i, j]
            if not np.isfinite(cij) or abs(cij) < 1e-12:
                continue
            
            for p in range(1, positions):
                add_quad(idx(i, p), idx(j, p + 1), cij)
    
    # Depot departure and return costs
    for i in range(n):
        lin[idx(i, 1)] += w[0, i]  # Start from depot
        lin[idx(i, positions)] += w[i, 0]  # Return to depot
    
    # Constraint 1: Exactly one location per position
    for p in range(1, positions + 1):
        qs = [idx(i, p) for i in range(n)]
        const += A * 1.0
        for q in qs:
            lin[q] += -A
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * A)
    
    # Constraint 2: Visit each city exactly once (skip depot city=0)
    for i in range(1, n):
        qs = [idx(i, p) for p in range(1, positions + 1)]
        const += B * 1.0
        for q in qs:
            lin[q] += -B
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                add_quad(qs[a], qs[b], 2.0 * B)
    
    # Convert to Pauli-Z form: x = (1-Z)/2
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
    
    # Build PennyLane Hamiltonian
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

def build_swap_mixer(num_cities):
    """
    SWAP mixer del artículo (Hamiltoniano de intercambio de orden),
    ADAPTADO para reemplazar build_xy_mixer directamente.
    """
    n = num_cities
    positions = n - 1

    coeffs = []
    ops = []

    # S+ = (X + iY)/2 , S- = (X - iY)/2
    choices_Splus  = [('X', 1.0), ('Y', 1.0j)]
    choices_Sminus = [('X', 1.0), ('Y', -1.0j)]
    pref = 1.0 / 16.0

    def pauli(ch, wire):
        if ch == 'X':
            return qml.PauliX(wire)
        if ch == 'Y':
            return qml.PauliY(wire)
        raise ValueError("Invalid Pauli")

    # H_mix = sum_{p=1}^{positions-1} sum_{u,v} H_PS(p,u,v)
    for p in range(1, positions):
        for u in range(n):
            for v in range(n):

                q_p_u   = idx(u, p, n)
                q_p_v   = idx(v, p, n)
                q_pp1_u = idx(u, p+1, n)
                q_pp1_v = idx(v, p+1, n)

                pauli_terms = {}

                # Expandir S+_p,u S+_{p+1,v} S-_p,v S-_{p+1,u}
                for (c1, k1) in choices_Splus:
                    for (c2, k2) in choices_Splus:
                        for (c3, k3) in choices_Sminus:
                            for (c4, k4) in choices_Sminus:

                                coef = pref * (k1*k2*k3*k4)

                                sparse = {
                                    q_p_u:   c1,
                                    q_pp1_v: c2,
                                    q_p_v:   c3,
                                    q_pp1_u: c4
                                }

                                key = tuple(sorted(sparse.items()))
                                pauli_terms[key] = pauli_terms.get(key, 0.0) + coef

                # Añadir término hermítico → 2 Re(...)
                for key, a in pauli_terms.items():
                    real_c = 2.0 * np.real(a)
                    if abs(real_c) < 1e-12:
                        continue

                    op = None
                    for (wire, ch) in key:
                        single = pauli(ch, wire)
                        op = single if op is None else (op @ single)

                    coeffs.append(real_c)
                    ops.append(op)

    return qml.Hamiltonian(coeffs, ops)

def build_x_mixer(num_cities):
    """Standard X mixer (allows constraint violations)"""
    num_qubits = num_cities * (num_cities - 1)
    terms = []
    coeffs = []
    
    for q in range(num_qubits):
        terms.append(qml.PauliX(q))
        coeffs.append(1.0)
    
    return qml.Hamiltonian(coeffs, terms)

# ============================================================================
# MIXER FACTORY
# ============================================================================

def build_mixer(mixer_type, num_cities):
    if mixer_type == 'x':
        return build_x_mixer(num_cities)
    elif mixer_type == 'xy':
        return build_xy_mixer(num_cities)
    elif mixer_type == 'swap':
        return build_swap_mixer(num_cities)
    else:
        raise ValueError(f"Mixer desconocido: {mixer_type}")

# ============================================================================
# CLASSICAL REFERENCE SOLUTION
# ============================================================================

print("\nREFERENCIA CLÁSICA (Fuerza Bruta):")
best_route_classic, best_cost_classic = best_classical_route(w)

route_names = [LOCATION_NAMES[r] for r in best_route_classic]
print(f"   Ruta óptima: {' → '.join(route_names)}")
print(f"   Costo óptimo: {best_cost_classic:.2f} km")

bitstring_warm = route_to_bitstring(best_route_classic, n)
print(f"\nBitstring warm-start: {bitstring_warm}")

# ============================================================================
# QUANTUM CIRCUIT SETUP
# ============================================================================

H_cost = build_cost_hamiltonian(w, A=A, B=B, C=C)
H_mixer = build_mixer(MIXER_TYPE, n)

print(f"\nHAMILTONIANO:")
print(f"H_cost terms: {len(H_cost.ops)}")
print(f"H_mixer terms: {len(H_mixer.ops)}")

dev = qml.device("default.qubit", wires=num_qubits)

def qaoa_layer(gamma, beta):
    """Single QAOA layer: cost evolution + mixer evolution"""
    qml.ApproxTimeEvolution(H_cost, gamma, 1)
    qml.ApproxTimeEvolution(H_mixer, beta, 1)

@qml.qnode(dev)
def energy_qaoa(params, warm=False):
    """QAOA circuit returning cost expectation value"""
    # Initial state
    if warm:
        # Warm start from classical solution
        for q_idx, bit in enumerate(bitstring_warm):
            if bit == '1':
                qml.PauliX(wires=q_idx)
    else:
        # Uniform superposition
        for q in range(num_qubits):
            qml.Hadamard(q)
    
    # QAOA layers
    for layer in range(p_layers):
        qaoa_layer(params[0][layer], params[1][layer])
    
    # Measure cost expectation
    return qml.expval(H_cost)

@qml.qnode(dev)
def probs_qnode(params, warm=False):
    """QAOA circuit returning full probability distribution"""
    # Initial state
    if warm:
        for q_idx, bit in enumerate(bitstring_warm):
            if bit == '1':
                qml.PauliX(wires=q_idx)
    else:
        for q in range(num_qubits):
            qml.Hadamard(q)
    
    # QAOA layers
    for layer in range(p_layers):
        qaoa_layer(params[0][layer], params[1][layer])
    
    return qml.probs(wires=range(num_qubits))

# ============================================================================
# OPTIMIZATION LOOP (Hybrid Quantum-Classical)
# ============================================================================

# Initialize parameters
rng = np.random.default_rng(42)
params = qml.numpy.array(
    rng.uniform(0, 2 * np.pi, (2, p_layers)),
    requires_grad=True
)

# Classical optimizer
opt = qml.AdamOptimizer(stepsize=0.05)

print("\n" + "=" * 70)
print(f" EJECUTANDO QAOA (p={p_layers}, warm_start={warm_start})")
print("=" * 70 + "\n")

for it in range(opt_steps):
    params = opt.step(lambda p: energy_qaoa(p, warm=warm_start), params)
    
    if it % 10 == 0 or it == opt_steps - 1:
        energy = energy_qaoa(params, warm=warm_start)
        print(f"Iter {it:4d} | Energy = {energy:.6f}")

print("\n Optimización completada     ")
print(f"   Parámetros óptimos: ({len(params[0])}, {len(params[1])})")

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

probs = np.array(probs_qnode(params, warm=warm_start))

top_k = 20
sorted_indices = np.argsort(probs)[::-1][:top_k]

print("\n" + "=" * 70)
print(f" TOP-{top_k} ESTADOS DECODIFICADOS")
print("=" * 70 + "\n")

candidate_solutions = []

for rank, idx_i in enumerate(sorted_indices, 1):
    bs = format(idx_i, f"0{num_qubits}b")
    route, valid, reason = decode_route(bs, n)
    prob = probs[idx_i]
    cost = classical_cost_of_route(route, w)
    
    # Translate route to location names
    route_names = []
    for r in route:
        if r is not None:
            route_names.append(LOCATION_NAMES[r])
        else:
            route_names.append("???")
    
    status = "✓" if valid else "✗"
    print(f"{rank:2d}. {status} [{bs}]")
    print(f"    Ruta: {' → '.join(route_names)}")
    print(f"    Prob={prob:.4f} | Costo={cost:.2f} km | Valid={valid}")
    
    if not valid:
        print(f"    Razón: {reason}")
    
    print()
    
    candidate_solutions.append((bs, route, valid, prob, cost, reason))

# ============================================================================
# FINAL RESULTS
# ============================================================================

valids = [c for c in candidate_solutions if c[2]]

print("\n" + "=" * 70)
print(" RESULTADOS FINALES")
print("=" * 70)
print(f"  Soluciones válidas: {len(valids)}/{top_k} ({100*len(valids)/top_k:.1f}%)")

if valids:
    best = min(valids, key=lambda x: x[4])
    best_route = best[1]
    best_cost = best[4]
    best_prob = best[3]
    
    route_names = [LOCATION_NAMES[r] for r in best_route if r is not None]
    
    print(f"\nMEJOR SOLUCIÓN VÁLIDA (QAOA):")
    print(f"Ruta: {' → '.join(route_names)}")
    print(f"Costo QAOA: {best_cost:.2f} km")
    print(f"Costo óptimo: {best_cost_classic:.2f} km")
    
    gap = ((best_cost - best_cost_classic) / best_cost_classic) * 100
    print(f"GAP: {gap:.2f}%")
    print(f"Probabilidad: {best_prob:.4f}")
    
    print()
    if gap <= 5:
        print("EXCELENTE: Gap <5% (mejor que D-Wave reportado)")
    elif gap <= 15:
        print("BUENO: Gap en rango esperado 5-15% según literatura")
    elif gap <= 25:
        print("ACEPTABLE: Gap 15-25%, considerar aumentar layers")
    else:
        print(f"ALTO: Gap >{gap:.1f}%")
    
    # Visualization
    if len(best_route) >= 2:
        clean_route = [r for r in best_route if r is not None]
        
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(w[i, j]):
                    G.add_edge(i, j, weight=float(w[i, j]))
        
        path_edges = [(clean_route[i], clean_route[i + 1]) 
                     for i in range(len(clean_route) - 1)]
        
        pos_layout = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(10, 8))
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos_layout, node_color='lightblue', 
                             node_size=800, alpha=0.9)
        
        # Draw labels with location names
        labels = {i: f"{i}\n{LOCATION_NAMES[i]}" for i in range(n)}
        nx.draw_networkx_labels(G, pos_layout, labels, font_size=10, 
                               font_weight='bold')
        
        # Draw all edges (background)
        nx.draw_networkx_edges(G, pos_layout, edge_color='lightgray', 
                             alpha=0.3, arrows=True, arrowsize=15)
        
        # Highlight optimal path
        nx.draw_networkx_edges(G, pos_layout, edgelist=path_edges, 
                             edge_color='red', width=3, arrows=True, 
                             arrowsize=20, arrowstyle='->')
        
        plt.title(f"Ruta Óptima QAOA - {DATASET.upper()}\n" + 
                 f"Costo: {best_cost:.2f} km | Gap: {gap:.2f}%",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"ruta_qaoa_{DATASET}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nGrafico guardado: {filename}")
        plt.show()
        plt.close()
    
    # Probability distribution
    top_labels = [f"#{i+1}" for i in range(len(sorted_indices))]
    top_probs = probs[sorted_indices]
    colors = ['green' if candidate_solutions[i][2] else 'red' 
             for i in range(len(sorted_indices))]
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_labels, top_probs, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Estado (ranking)', fontsize=12)
    plt.ylabel('Probabilidad', fontsize=12)
    plt.title(f'Top-{top_k} Estados - {DATASET.upper()}\n' + 
             'Verde=Válido, Rojo=Inválido',
             fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = f"probabilidades_qaoa_{DATASET}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Grafico guardado: {filename}")
    plt.show()
    plt.close()

else:
    print("\nNO HAY SOLUCIONES VALIDAS")
    print("   Accion: Aumentar A, B a 150 o mas")
    print("No se genero visualizacion (sin solucion valida)")

print("\n" + "=" * 70)
print(" EJECUCION COMPLETADA")
print("=" * 70)
