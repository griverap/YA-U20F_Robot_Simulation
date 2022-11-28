import numpy as np
from copy import copy
import os
 
pi = np.pi
 
## Creacion de carpeta de guardado tmp
try:
    os.mkdir("/home/user/lab_ws/src/cinematica_proyecto/tmp")
except OSError as error:
    print(error)
 
# Files for the logs (Newton y Gradiente)
fq_n = open("/home/user/lab_ws/src/cinematica_proyecto/tmp/qcurrent_n.txt", "w")
ferror_n = open("/home/user/lab_ws/src/cinematica_proyecto/tmp/error_n.txt", "w")
fq_g = open("/home/user/lab_ws/src/cinematica_proyecto/tmp/qcurrent_g.txt", "w")
ferror_g = open("/home/user/lab_ws/src/cinematica_proyecto/tmp/error_g.txt", "w")
 
def dh(theta, d, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
 
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T
 
 
def fkine(q):
    """
    Calcular la cinematica directa del robot dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6, q7]
 
    """
    # Longitudes (en metros)
    l1=0.41
    l2=0.49
    l3=0.42
    l4=0.18
 
    T01 = dh( pi+q[0] , l1, 0 , pi/2 )
    T12 = dh( pi+q[1] , 0 , 0 , pi/2 )
    T23 = dh( q[2] , l2, 0 , pi/2 )
    T34 = dh( pi+q[3] , 0 , 0 , pi/2 )
    T45 = dh( q[4] , l3, 0 , pi/2 ) 
    T56 = dh( pi+q[5] , 0 , 0 , pi/2 )
    T67 = dh( q[6]  , l4, 0 ,   0  )
 
 
    # Efector final con respecto a la base
    T = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(T67)
    return T
 
 
def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x7 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6 , q7]
 
    """
    # Alocacion de memoria
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine(q)
    # Iteracion para la derivada de cada columna
    for i in xrange(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i]+delta
        # Transformacion homogenea luego del incremento (q+dq)
        Ti = fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0:3, i] = (Ti[0:3, 3] - T[0:3, 3])/delta
    return J
 
 
def ikine_robot(xdes, q0):
    """
    Calcular la cinematica inversa del robot numericamente a partir de la configuracion articular inicial de q0. 
    Metodo de newton
    """
 
    # Parametros
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
 
    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q, delta)
        T = fkine(q)
        f = T[0:3, 3]
        # error
        e = xdes - f
        # actualizacion de q
        q = q + np.dot(np.linalg.pinv(J), e)
 
        # guardado de valores para q y error
        ferror_n.write(str(e[0])+' '+str(e[1])+' '+str(e[2])+'\n')
        fq_n.write(str(q[0])+" "+str(q[1])+" "+str(q[2])+" "+str(q[3])+" "+
             str(q[4])+" "+str(q[5])+" "+str(q[6])+"\n")
 
        # Condicion de cierre
        if (np.linalg.norm(e) < epsilon):
            break
    return q
 
def ik_gradiente(xdes, q0):
    """
    Calcular la cinematica inversa  numericamente a partir de la configuracion articular inicial de q0. 
    Metodo gradiente
    """
 
    # Parametros
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    alfa     = 0.5
 
    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q, delta)
        T = fkine(q)
        f = T[0:3, 3]
        # error
        e = xdes - f
        # actualizacion de q
        q = q + alfa*np.dot(J.T, e)
        
        # guardado de valores para q y error
        ferror_g.write(str(e[0])+' '+str(e[1])+' '+str(e[2])+'\n')
        fq_g.write(str(q[0])+" "+str(q[1])+" "+str(q[2])+" "+str(q[3])+" "+
             str(q[4])+" "+str(q[5])+" "+str(q[6])+"\n")
 
        # Condicion de cierre
        if (np.linalg.norm(e) < epsilon):
            break
    return q
