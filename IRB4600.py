import RemoteAPIs.vrep as vrep
import math
import numpy
import random
import botControl


class IRB4600(botControl.BoTControl):
    def __init__(self, bot_name, joint_name, joint_num):
        mode = vrep.simx_opmode_blocking
        super(IRB4600, self).__init__(bot_name, joint_name, joint_num)
        err0 = self.connect_bot('127.0.0.1', 19999)
        # err1, self.ik_tip = vrep.simxGetObjectHandle(self.client_id,
        #                                             'IRB4600_IkTip', mode)
        # err2, self.ikTarget = vrep.simxGetObjectHandle(self.client_id,
        #                                               'IRB4600_IkTarget', mode)
        # err3, self.aux_joint = vrep.simxGetObjectHandle(self.client_id,
        #                                                'IRB4600_auxJoint', mode)
        # print(err0, err1, err2, err3)

    def set_ik_mode(self, mode=vrep.simx_opmode_oneshot):
        err0 = vrep.simxSetObjectParent(self.client_id, self.ikTarget,
                                        self.ik_tip, True, mode)
        err1 = vrep.simxSetObjectPosition(self.client_id, self.ikTarget,
                                          self.ik_tip, [0, 0, 0], mode)
        err2 = vrep.simxSetObjectPosition(self.client_id, self.ikTarget,
                                          self.ik_tip, [0, 0, 0], mode)
        print(err0, err1, err2)

    def simple_test(self):
        self.set_joint_position([90*math.pi/180,
                                 -30*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180
                                 ])
        self.set_joint_position([0*math.pi/180,
                                 -30*math.pi/180,
                                 60 * math.pi/180,
                                 0*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180])
        self.set_joint_position([90*math.pi/180,
                                 -30*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180
                                 ])
        self.set_joint_position([90*math.pi/180,
                                 -30*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180,
                                 60*math.pi/180,
                                 0*math.pi/180
                                 ])


IRB = IRB4600('IRB4600', 'IRB4600_joint', 6)
while True:
    IRB.simple_test()
