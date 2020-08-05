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
        self.joint_handle = np.zeros((joint_num,))
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
        if self.connect(ip, port) != -1:
            ret_code, self.bot = vrep.simxGetObjectHandle(
                self.client_id, self.bot_name, vrep.simx_opmode_blocking)
            return 0 if(ret_code == vrep.simx_return_ok) else -1
        return -1

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
            # print(self.joint_handle[i])
            _, joint_pos = vrep.simxGetJointPosition(self.client_id,
                                                     int(self.joint_handle[i]),
                                                     mode)
            #self.joint_pos[i] = round((joint_pos*self.RAD2EDG), 2)
            self.joint_pos[i] = joint_pos
        print(self.joint_handle[i], " : ", self.joint_pos)

        return self.joint_pos

    def set_joint_position(self, position=None, mode=vrep.simx_opmode_oneshot):
        self.get_joint_position()
        # sync pasuse
        vrep.simxPauseCommunication(self.client_id, True)
        for i in range(self.joint_num):
            # test
            if int(self.simu_time) % 2 == 0:
                rand = random.uniform(0.1, 0.4)
            else:
                rand = random.uniform(-0.1, 0.4)

            vrep.simxSetJointPosition(self.client_id,
                                      int(self.joint_handle[i]),
                                      float(self.joint_pos[i])+rand,
                                      mode)
        vrep.simxPauseCommunication(self.client_id, False)

    def simple_test(self,):
        '''

        :param current_joint:
        :param desired_joint:
        :return:
        '''
        self.get_joint_handle()
        # self.get_joint_position()
        while True:
            self.simu_time = self.simu_time + self.tstep
            self.set_joint_position()


boTControl = BoTControl('IRB4600', 'IRB4600_joint', 6)
err = boTControl.connect_bot('127.0.0.1', 19999)
boTControl.simple_test()
