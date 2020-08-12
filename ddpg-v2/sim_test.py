import vrep
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server')

       # setup a useless signal
    vrep.simxSetIntegerSignal(clientID, 'line', 1, vrep.simx_opmode_oneshot)

    for j in range(5):
        print('---------------------simulation', j)

        # IMPORTANT
        # you should poll the server state to make sure
        # the simulation completely stops before starting a new one
        while True:
            # poll the useless signal (to receive a message from server)
            vrep.simxGetIntegerSignal(
                clientID, 'line', vrep.simx_opmode_oneshot)

            # check server state (within the received message)
            e = vrep.simxGetInMessageInfo(clientID,
                                          vrep.simx_headeroffset_server_state)

            # check bit0
            not_stopped = e[1] & 1

            if not not_stopped:
                break
            else:
                print('not_stopped')
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

        # IMPORTANT
        # you should always call simxSynchronous()
        # before simxStartSimulation()
        vrep.simxSynchronous(clientID, True)
        # then start the simulation:
        e = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        print('start', e)

        # Now step a few times:
        for i in range(2):
            e = vrep.simxSynchronousTrigger(clientID)
            print('synct', e)
            # wait till simulation step finish
            e = vrep.simxGetPingTime(clientID)
            print('getping', e)

        # stop the simulation:
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        print('stop')

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
