import sys
sys.path.append('../')
import RemoteAPIs.vrep as vrep
import time
import numpy as np
import  time
tstep = 0.005
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    print('Connected to remote API failed!')


def get_shape_position(name='myCuboid'):
    err1, handle1 = vrep.simxGetObjectHandle(clientID, name,
                                             vrep.simx_opmode_blocking)
    for i in range(1, 10):
        pos = vrep.simxGetObjectPosition(clientID, handle1, -1,
                                         vrep.simx_opmode_streaming)
        # record time
        print(pos)
        time.sleep(0.1)

def stream():
    # prerequisite of retrieving the data from the buffer: declartion the mode in streaming
    error, next_angle = vrep.simxGetIntegerSignal(clientID, "joint1",
                                                vrep.simx_opmode_streaming)
def get_arm_joint(handle_name='IRB140', arm_name='arm_joint', joint_num=6):
    arm_joint = np.zeros((joint_num,), dtype=np.int)
    # reset
    err, temp_joint = vrep.simxGetFloatSignal(clientID,
                                              "joint1",
                                              vrep.simx_opmode_streaming)
    #stream()
    for i in range(0, 10):
        # get joint position
        """
        for i in range(1, 7):
            joint_err, temp_joint = vrep.simxGetStringSignal(clientID,
                                                             arm_name+str(i),
                                                             vrep.simx_opmode_blocking)
            print(arm_name+str(i))
            print(joint_err, temp_joint)
        """
        # important ! you need add this to trigger v-rep sync
        vrep.simxSynchronousTrigger(clientID)
        #vrep.simxGetPingTime(clientID)
        err, temp_joint = vrep.simxGetFloatSignal(clientID,
                                                  "joint1",
                                                  vrep.simx_opmode_buffer)
        print(err, temp_joint)
        time.sleep(0.2)

while True:
    e = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
    not_stopped = e[1] & 1
    if not not_stopped:
        break
    else:
        print('not stopped')
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

vrep.simxSynchronous(clientID, True)
e = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
print('start', e)
get_arm_joint()
vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
