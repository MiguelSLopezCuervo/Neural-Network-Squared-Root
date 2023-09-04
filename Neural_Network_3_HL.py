# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 13:57:05 2023

@author: Miguel Sánchez

EN ESTE CASO SENCILLO SE VE QUE ES MUCHO MEJOR UNA RED NEURONAL DE 2 
HIDDEN LAYERS, PORQUE FUNCIONA MUCHO MEJOR.

NO OBSTANTE ES MUY POSIBLE QUE HAYA UNA COMBINACIÓN DE HIPERPARÁMETROS MUCHO
MEJOR QUE ESTA Y QUE CONSIGAN MEJORES RESULTADOS QUE LA RED NEURONAL DE 2
HIDDEN LAYERS
"""

import numpy as np
import time
import Funciones_Neural_Network_3HL as fun

m = 2000
m_test = 100
it = 50000
dim_red = [1, 5, 5, 5, 1] #OJO, LA PRIMERA ES EL INPUT, ES N.N. DE 3 CAPAS (2 hidden)
learning_rate = 0.00002

# Registra el tiempo de inicio
inicio = time.time()

# Comenzamos creando el dataset con el que alimentar a la red
X, Y = fun.CrearDataSet(m, "raiz")

# Para hacer la red PRIMERO SE INICIALIZAN LOS PARÁMETROS
parameters = fun.IniciarParametrosNetwork(dim_red) #CHECK (funciona)

# A continuación se entrena mediante descenso del gradiente
for i in range(0, it):
    #SE REALIZA LA FORWARD PROPAGATION
    A, cache_Z, cache_A = fun.Forward_Propagation(X, parameters, dim_red)
    #Se actualizan los parámetros mediante back propagation
    parameters = fun.UpdateParameters(Y, parameters, cache_Z, cache_A, m, dim_red, learning_rate)
    #Señas de vida
    if i%5000==0:
        # Registra el tiempo de finalización
        final = time.time()
        print("Iteración: ", i, "Coste:", fun.Coste(Y, A, m), "tiempo", (final-inicio)/60)

# Una vez entrenada la red, vemos como se comporta con casos nuevos
X_test, Y_test = fun.CrearDataSet(m_test, "raiz")
Y_pred , bs1, bs2 = fun.Forward_Propagation(X_test, parameters, dim_red)
fun.Graficar(X_test, Y_test , Y_pred)
print("Coste del test: ", fun.Coste(Y_test, Y_pred, m_test))

    
    
    