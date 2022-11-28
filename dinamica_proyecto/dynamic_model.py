#!/usr/bin/env python

import rbdl
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Lectura del modelo del robot a partir de URDF (parsing)
modelo = rbdl.loadModel(
    '/home/user/lab_ws/src/fproject_description/urdf/fproject.urdf')
# Grados de libertad
ndof = modelo.q_size

# Configuracion articular
q = np.array([1.1, 1., 0., 0.7, 0.5, 0.3, 0.2])
# Velocidad articular
dq = np.array([0.4, 1., 0., 0.6, 0.8, 0.2, 0.1])
# Aceleracion articular
ddq = np.array([0.1, 0., 1, 0.5, 0., 0.5, 0.])

# Arrays numpy
zeros = np.zeros(ndof)          # Vector de ceros
tau = np.zeros(ndof)          # Para torque
g = np.zeros(ndof)          # Para la gravedad
c = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
M = np.zeros([ndof, ndof])  # Para la matriz de inercia
e = np.eye(ndof)               # Vector identidad

print(ndof)
# Torque dada la configuracion del robot
rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
print("Vector torque: \n{}".format(tau))

# Vector gravedad
rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
print("Vector gravedad: \n{}".format(g))

# Vector coriolis
rbdl.InverseDynamics(modelo, q, dq, zeros, c)
c = c - g
print("Vector coriolis: \n{}".format(c))

# Matriz inercia
for i in range(ndof):
    rbdl.InverseDynamics(modelo, q, zeros, e[i, :], M[i, :])
    M[i, :] = M[i, :] - g
print("Matriz inercia: \n{}".format(M))
