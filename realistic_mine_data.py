import numpy as np

#Config
N_Locations = 8
N_Steps=8

#Ubicaciones example
locations = {
    0: "Deposito",
    1: "Frente_Norte",
    2: "Frente_Noroeste",
    3: "Frente_Centro",
    4: "Frente_Sur",
    5: "Frente_Sureste",
    6: "Botadero_Principal",
    7: "Botadero_Secundario",
}

#Matriz distancias
#Entre Frentes: 3-5km
#Frentes a Botaderos: 3-5km

distance_matrix = np.array([
    #  0    1    2    3    4    5    6    7
    [0.0, 2.5, 3.2, 4.1, 4.8, 5.3, 6.5, 7.2],  # Depósito
    [2.5, 0.0, 3.5, 4.2, 6.1, 6.8, 5.8, 7.5],  # Frente_Norte
    [3.2, 3.5, 0.0, 3.8, 5.4, 6.2, 6.2, 8.1],  # Frente_Noreste
    [4.1, 4.2, 3.8, 0.0, 4.1, 4.8, 7.3, 8.9],  # Frente_Centro
    [4.8, 6.1, 5.4, 4.1, 0.0, 3.6, 8.5, 9.2],  # Frente_Sur
    [5.3, 6.8, 6.2, 4.8, 3.6, 0.0, 9.1, 8.7],  # Frente_Sureste
    [6.5, 5.8, 6.2, 7.3, 8.5, 9.1, 0.0, 4.5],  # Botadero_Principal
    [7.2, 7.5, 8.1, 8.9, 9.2, 8.7, 4.5, 0.0],  # Botadero_Secundario
])

# Rutas bidireccionales (automatización de rutas)
assert np.allclose(distance_matrix, distance_matrix.T), "La matriz de distancias no es simétrica"

# Rangos
frentes = [1, 2, 3, 4, 5]
botaderos = [6, 7]

# Distancia mínima entre frentes
for i in frentes:
    for j in frentes:
        if i < j:
            dist_ij = distance_matrix[i, j]
            assert 3.0 <= dist_ij <= 6.8, "La distancia entre frentes no está en el rango permitido"

# Distancia mínima entre frentes y botaderos  
for i in frentes:
    for j in botaderos:
        dist_ij = distance_matrix[i, j]
        assert 5.0 <= dist_ij <= 10.0, "La distancia entre frentes y botaderos no está en el rango permitido"

# Distancia mínima entre botaderos
for i in botaderos:
    for j in botaderos:
        if i < j:
            dist_ij = distance_matrix[i, j]
            assert 3.0 <= dist_ij <= 6.5, "La distancia entre botaderos no está en el rango permitido"

print("Matriz info:")
print("N frentes de carga: ", len(frentes))
print("N botaderos: ", len(botaderos))
print("N ubicaciones: ", N_Locations)
print("N qubits: ", N_Locations * N_Steps, "=", N_Locations**2)
