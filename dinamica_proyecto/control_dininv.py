#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages

import os
import rbdl


rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
try:
    os.mkdir("/home/user/lab_ws/src/dinamica_proyecto/control_diniv_data")
except OSError as error:
    print(error)

fqact = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_diniv_data/qactual.txt", "w")
fqdes = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_diniv_data/qdeseado.txt", "w")
fxact = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_diniv_data/xactual.txt", "w")
fxdes = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_diniv_data/xdeseado.txt", "w")

# Nombres de las articulaciones
jnames = ['joint_1', 'joint_2', 'joint_3',
          'joint_4', 'joint_5', 'joint_6', 'joint_7']
# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([1.0, -1.0, 1.0, 1.3, -1.5, 1.0, 0.5])
# Velocidad articular deseada
dqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Aceleracion articular deseada
ddqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine(qdes)[0:3, 3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel(
    '/home/user/lab_ws/src/fproject_description/urdf/fproject.urdf')
ndof = modelo.q_size     # Grados de libertad
zeros = np.zeros(ndof)     # Vector de ceros

# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Bucle de ejecucion continua
t = 0.0

# Se definen las ganancias del controlador
valores = 0.1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

while not rospy.is_shutdown():

    # Leer valores del simulador
    q = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine(q)[0:3, 3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' ' +
                str(q[2])+' ' + str(q[3])+' '+str(q[4])+' '+str(q[5])+' '+str(q[6])+'\n')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' ' + str(
        qdes[2])+' ' + str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+' '+str(qdes[6])+'\n')

    # ----------------------------
    # Control dinamico (COMPLETAR)
    # ----------------------------
# Vector de ceros
    zeros = np.zeros(ndof)

    # Vector para torque
    tau = np.zeros(ndof)

    # Vector para gravedad
    g = np.zeros(ndof)

    # Vector para coriolis y centrifuga
    c = np.zeros(ndof)

    # Matriz para la inercia
    M = np.zeros([ndof, ndof])

    # Vector identidad
    e = np.eye(7)

    # Hallando el valor del torque
    rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

    # Hallando el valor de la gravedad
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)

    # Hallando la coriolis y centrifuga
    rbdl.InverseDynamics(modelo, q, dq, zeros, c)
    c = c - g

    # Hallando matriz de inercia
    for i in range(ndof):
        rbdl.InverseDynamics(modelo, q, zeros, e[i, :], M[i, :])
        M[i, :] = M[i, :] - g

    # Dinamica de error de segundo orden
    # Calculo error posicion
    e = qdes - q
    # Calculo error velocidad
    de = dqdes - dq
    # Error del Kd
    Kde = Kd.dot(de)

    # Hallando torque
    u = M.dot(ddqdes + Kde + Kp.dot(e)) + c.dot(dq) + g

    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()
