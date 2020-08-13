import sys
sys.path.append('../')
import RemoteAPIs.vrep as vrep
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
vrep.simxSynchronous(clientID, True)
print(clientID)
def get_shape_position(name):
    err1, handle1 = vrep.simxGetObjectHandle(clientID, name,
                                             vrep.simx_opmode_blocking)
    err2, handle2 = vrep.simxGetObjectHandle(clientID, 'IRB140_joint1',
                                             vrep.simx_opmode_blocking)
    #vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

    pos = vrep.simxGetObjectPosition(clientID, handle2, -1,
                                     vrep.simx_opmode_streaming)
    print(pos)


get_shape_position('Cuboid')
