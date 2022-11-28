import numpy as np
from copy import copy
import rbdl

pi = np.pi


class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel(
            '/home/user/lab_ws/src/fproject_description/urdf/fproject.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq


def dh(d, theta, a, alpha):
    """
    Matriz de transformacion homogenea asociada a los parametros DH.
    Retorna una matriz 4x4
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine(q):
    # Matrices DH
    l1 = 0.41
    l2 = 0.49
    l3 = 0.42
    l4 = 0.18

    T01 = dh(pi+q[0], l1, 0, pi/2)
    T12 = dh(pi+q[1], 0, 0, pi/2)
    T23 = dh(q[2], l2, 0, pi/2)
    T34 = dh(pi+q[3], 0, 0, pi/2)
    T45 = dh(q[4], l3, 0, pi/2)
    T56 = dh(pi+q[5], 0, 0, pi/2)
    T67 = dh(q[6], l4, 0,   0)

    # Efector final con respecto a la base
    T = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(T67)
    return T


def jacobian_position(q, delta=0.0001):
    # Alocacion de memoria
    J = np.zeros((3, 7))
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
