import RemoteAPIs.vrep as vrep
import math
import numpy as np
import random


class BoTControl(object):
    def __init__(self, bot_name, joint_name, joint_num):
        print('init control')
        vrep.simxFinish(-1)  # close all open connected
        self.client_id = 0
        self.bot = 0
        self.RAD2EDG = 180/math.pi
        self.tstep = 0.005
        self.bot_name = bot_name
        self.joint_num = joint_num
        self.joint_name = joint_name
        self.joint_handle = np.zeros((joint_num,), dtype=np.int)
        self.joint_pos = np.zeros((joint_num,))
        self.simu_time = 0

    def connect(self, ip, port):
        '''
        Connect to server
        :param ip:
        :param port:
        :return:
        '''
        self.client_id = vrep.simxStart(ip, port, True, True, 5000, 5)
        print('connect id', self.client_id)
        # sync
        vrep.simxSetFloatingParameter(self.client_id,
                                      vrep.sim_floatparam_simulation_time_step,
                                      self.tstep,
                                      vrep.simx_opmode_oneshot)
        # open sync mode
        vrep.simxSynchronous(self.client_id, True)
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot)

        return self.client_id if self.client_id != -1 else -1

    def connect_bot(self, ip, port):
        '''
        Connect to Bot
        :param ip:
        :param port:
        :param botName:
        :return:
        '''
        self.get_joint_handle()
        ret_code, self.bot = vrep.simxGetObjectHandle(self.client_id,
                                                      self.bot_name,
                                                      vrep.simx_opmode_blocking)
        print('connect bot result', ret_code)

    def get_joint_handle(self, mode=vrep.simx_opmode_blocking):
        print('get joint handle')
        for i in range(self.joint_num):

            _, joint_handle = vrep.simxGetObjectHandle(self.client_id,
                                                       self.joint_name +
                                                       str(i+1),
                                                       mode)
            print(self.joint_name+str(i+1), " : ", joint_handle)
            self.joint_handle[i] = joint_handle
        return self.joint_handle

    def get_joint_position(self, mode=vrep.simx_opmode_streaming):
        for i in range(self.joint_num):

            err, j_pos = vrep.simxGetJointPosition(self.client_id,
                                                   self.joint_handle[i],
                                                   mode)
            print(err, round((j_pos*self.RAD2EDG), 2))
            self.joint_pos[i] = j_pos

        return self.joint_pos

    def set_joint_position(self, mode=vrep.simx_opmode_oneshot):
        # sync pasuse
        vrep.simxPauseCommunication(self.client_id, True)
        for i in range(self.joint_num):
            vrep.simxSetJointPosition(self.client_id,
                                      self.joint_handle[i],
                                      120/self.RAD2EDG,
                                      mode)
        vrep.simxPauseCommunication(self.client_id, False)

    def simple_test(self,):
        '''

        :param current_joint:
        :param desired_joint:
        :return:
        '''
        last_cmd_time = vrep.simxGetLastCmdTime(self.client_id)
        vrep.simxSynchronousTrigger(self.client_id)
        while vrep.simxGetConnectionId(self.client_id) != -1:
            curr_cmd_time = vrep.simxGetLastCmdTime(self.client_id)
            dt = curr_cmd_time - last_cmd_time

            self.get_joint_position()
            self.set_joint_position()

            last_cmd_time = curr_cmd_time
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)


boTControl = BoTControl('IRB4600', 'IRB4600', 6)
err = boTControl.connect('127.0.0.1', 19999)
err = boTControl.connect_bot('127.0.0.1', 19999)
boTControl.get_joint_position()
boTControl.simple_test()
