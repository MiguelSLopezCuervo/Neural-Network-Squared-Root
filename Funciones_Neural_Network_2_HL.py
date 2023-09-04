# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 13:57:43 2023

@author: Miguel Sánchez
"""

import numpy as np
import matplotlib.pyplot as plt

"""
CREAMOS PRIMERO EL DATASET, SEGÚN LO QUE QUERAMOS CALCULAR SERÁ UN DATASET
U OTRO
"""
def CrearDataSet(m, tipo):
    X = np.random.rand(1, m) * 10

    if tipo == "comb lin":
        Y = 3 * X + 1
    if tipo == "raiz":
        Y = np.sqrt(X)
    if tipo == "log":
        Y = np.log(X)
        
    return X, Y

"""
FORWARD PROPAGATION, IMPLICA INICIAR PARÁMETROS, DEFINIR LA FUNCIÓN DE
ACTIVACIÓN (QUE VA A SER RELU), Y EL COSTE FINAL 
"""
def IniciarParametrosNetwork(dim_red):
    L = len(dim_red)
    parameters = {}
    for i in range(1, L):
        parameters["W" + str(i)]=np.random.randn(dim_red[i], dim_red[i-1]) #Elijo esos para que salgan la mayoría positivos y valores centrados en 0.5
        parameters["b" + str(i)] = np.zeros((dim_red[i], 1))
    return parameters
    
def ReLU(Z):
    A = np.copy(Z)
    # Aplica ReLU de manera óptima
    A[A < 0] = 0
    return A
def Der_ReLU(Z):
    RES = np.copy(Z)
    # Aplica ReLU de manera óptima
    RES[RES < 0] = 0
    RES[RES >= 0] = 1
    return RES

def Forward_Propagation(X, parameters, dim_red):
    A = np.copy(X)
    cache_Z = []
    cache_A = [X]
    L = len(dim_red)
    for i in range(1, L):
        Z = np.dot(parameters["W"+ str(i)], A) + parameters["b"+ str(i)]
        A = ReLU(Z)
        cache_Z.append(Z)
        cache_A.append(A)
    return A, cache_Z, cache_A

def Coste(Y, Y_hat, m): #Coste uso MSE (para la binary clasification no valía, para esta aplicación sí)
    J = 1/m * np.sum((Y - Y_hat) ** 2)
    return J

"""
ACTUALIZAR PARÁMETROS, QUE IMPLICA BACK PROPAGATION
"""
def UpdateParameters(Y, parameters, cache_Z, cache_A, m, dim_red, learning_rate=0.001):
    dparameters = Back_Propagation(Y, parameters, cache_Z, cache_A, m, dim_red)
    L = len(dim_red)
    for i in range(1, L):
        parameters["W"+ str(i)] -= learning_rate * dparameters["dW"+ str(i)]
        parameters["b" + str(i)] -= learning_rate * dparameters["db" + str(i)]
    return parameters

def Back_Propagation(Y, parameters, cache_Z, cache_A, m, dim_red):
    L = len(dim_red)
    dA_prev = 2 * (cache_A[L-1]-Y)
    dparameters = {}
    for i in range(L-1, 0, -1):
        dZ = dA_prev * Der_ReLU(cache_Z[i-1])
        dparameters["dW"+str(i)] = 1/m * np.dot(dZ, cache_A[i-1].T)
        dparameters["db"+str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(parameters["W"+str(i)].T, dZ)    
    return dparameters
"""
PARA GRAFICAR
"""
def Graficar(X_test, Y_test, Y_pred):
    # Crear la gráfica
    plt.figure(figsize=(8, 6))
    
    # Graficar los puntos de Y_test en la coordenada X correspondiente de X_test
    plt.plot(X_test, Y_test, label='Y_test', marker='o', linestyle='-', color='blue', linewidth=2)
    
    # Graficar los puntos de Y_pred en la coordenada X correspondiente de X_test
    plt.plot(X_test, Y_pred, label='Y_pred', marker='x', linestyle='-', color='red', linewidth=2)
    
    plt.xlabel('Valores de X_test')
    plt.ylabel('Valores de Y')
    plt.title('Gráfica de Y_test y Y_pred 2 hidden layers')
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()
    
    
    
    