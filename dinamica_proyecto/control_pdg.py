#!/usr/bin/env python

import os
import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages

import rbdl

rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
# Archivos donde se almacenara los datos
try:
    os.mkdir("/home/user/lab_ws/src/dinamica_proyecto/control_pdg_data")
except OSError as error:
    print(error)

fqact = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_pdg_data/qactual.txt", "w")
fqdes = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_pdg_data/qdeseado.txt", "w")
fxact = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_pdg_data/xactual.txt", "w")
fxdes = open(
    "/home/user/lab_ws/src/dinamica_proyecto/control_pdg_data/xdeseado.txt", "w")

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
# Configuracion articular deseada
qdes = np.array([1.0, -1.0, - 0.2, 0.2, 0.8, -0.8, 0.5])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine(qdes)[0:3, 3]
# Green marker shows the desired position
bmarker_deseado.xyz(xdes)
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel(
    '/home/user/lab_ws/src/fproject_description/urdf/fproject.urdf')
ndof = modelo.q_size     # Grados de libertad

# Frecuencia del envio (en Hz)
freq = 200
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Se definen las ganancias del controlador
valores = 0.1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

# Bucle de ejecucion continua
t = 0.0
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
    # Peso por gravedad
    gravity = np.zeros(ndof)
    rbdl.InverseDynamics(modelo, q, zeros, zeros, gravity)
    # u = np.zeros(ndof)   # Reemplazar por la ley de control
    # Calculo de Tau
    u = gravity + Kp.dot(qdes - q) - Kd.dot(dq)

    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)

    bmarker_actual.xyz(x)
    bmarker_deseado.publish()
    bmarker_actual.publish()
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()
