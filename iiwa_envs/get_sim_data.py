import pybullet as p
import time
import numpy as np
import pybullet_data
import os
import rosbag
from robots import __file__ as path_robots
from scipy import signal
import matplotlib.pyplot as plt

class Model:
    def __init__(self, hyperparams):

        self.res0 = 0.2       # res0
        self.res1 = hyperparams[1]     # res1
        self.res2 = hyperparams[1]     # res2
        # self.res3 = hyperparams[3]     # res3
        self.res4 = hyperparams[1]     # res4
        self.res5 = hyperparams[1]     # res5
        # self.res6 = hyperparams[6]     # res6
        self.res7 = hyperparams[1]     # res7
        self.res8 = hyperparams[1]     # res58
        self.latf = hyperparams[0]       # latf
        self.angvel = hyperparams[2]
        # self.id = id
    def read_bag(self, bag):
        mallet_poses = []
        puck_poses = []
        table_poses = []
        count_table = 0
        count_puck = 0
        count_mallet = 0
        for topic, msg, t in bag.read_messages():

            t_start = bag.get_start_time()
            # print(t_start)
            t_end = bag.get_end_time()
            t_i = t.to_sec() - t_start
            if topic == '/tf':
                if msg.transforms[0].child_frame_id == "Mallet":
                    count_mallet += 1
                    pose_i = np.array([msg.transforms[0].transform.translation.x,
                                       msg.transforms[0].transform.translation.y,
                                       msg.transforms[0].transform.translation.z,
                                       msg.transforms[0].transform.rotation.x,
                                       msg.transforms[0].transform.rotation.y,
                                       msg.transforms[0].transform.rotation.z,
                                       msg.transforms[0].transform.rotation.w,
                                       t_i])
                    if len(mallet_poses) == 0 or not np.equal(np.linalg.norm(mallet_poses[-1][:2] - pose_i[:2]), 0):
                        mallet_poses.append(pose_i)

                elif msg.transforms[0].child_frame_id == "Puck":
                    count_puck += 1
                    pose_i = np.array([msg.transforms[0].transform.translation.x,
                                       msg.transforms[0].transform.translation.y,
                                       msg.transforms[0].transform.translation.z,
                                       msg.transforms[0].transform.rotation.x,
                                       msg.transforms[0].transform.rotation.y,
                                       msg.transforms[0].transform.rotation.z,
                                       msg.transforms[0].transform.rotation.w,
                                       t_i])
                    if len(puck_poses) == 0 or not np.equal(np.linalg.norm(puck_poses[-1][:2] - pose_i[:2]), 0):
                        puck_poses.append(pose_i)

                elif msg.transforms[0].child_frame_id == "Table":
                    count_table += 1
                    pose_i = np.array([msg.transforms[0].transform.translation.x,
                                       msg.transforms[0].transform.translation.y,
                                       msg.transforms[0].transform.translation.z,
                                       msg.transforms[0].transform.rotation.x,
                                       msg.transforms[0].transform.rotation.y,
                                       msg.transforms[0].transform.rotation.z,
                                       msg.transforms[0].transform.rotation.w,
                                       t_i])
                    if len(table_poses) == 0 or not np.equal(np.linalg.norm(table_poses[-1][2] - pose_i[2]), 0):
                        table_poses.append(pose_i)


        mallet_poses = np.array(mallet_poses)
        puck_poses = np.array(puck_poses)
        table_poses = np.array(table_poses)

        return puck_poses, mallet_poses, table_poses, t_end - t_start

    def collision_filter(self, puck, table):
        p.setCollisionFilterPair(puck, table, 0, 0, 1)
        p.setCollisionFilterPair(puck, table, 0, 1, 1)
        p.setCollisionFilterPair(puck, table, 0, 2, 1)
        p.setCollisionFilterPair(puck, table, 0, 4, 1)
        p.setCollisionFilterPair(puck, table, 0, 5, 1)
        p.setCollisionFilterPair(puck, table, 0, 6, 1)
        p.setCollisionFilterPair(puck, table, 0, 7, 1)

# relations
# slide: t_base (0)
# restitution and friction
#       t_down_rim_l (1),
#       t_down_rim_r (2),
#       t_left_rim (4),
#       t_right_rim (5),
#       t_up_rim_l (6),
#       t_up_rim_r (7)


    def filter(self, data, Wn=0.1):
        b, a = signal.butter(3, Wn)  # Wn is the frequency when amplitude drop of 3 dB, smaller Wn, smaller frequency pass

        for i in range(data.shape[1]):
            data[:, i] = signal.filtfilt(b, a, data[:, i], method='gust')
        return data

    def Diff(self, data, h, Wn=0.3):
        slope = np.zeros((data.shape))
        # data_ = np.zeros((data.shape[0], 2))

    #   data changed after using, so .copy is necessary
    #   tradeoff: large Wn, good coincidence with original traj, position noised. small Wn, bad coincidence with ori. traj at corner, position less noised
    #   in order to get initial velocity, here ignore the corer precision with small Wn
    #     data = filter(data.copy(), 0.7)
        t_stueck = self.t / data.shape[0]
        for i in range(data.shape[0]):  # Differenzenquotienten
            if i > data.shape[0] - h - 1:
                slope[i, :] = np.zeros((1, data.shape[1]))
                break
            slope[i, :] = (data[i + h, :] - data[i, :]) / (h * t_stueck)
        slope = self.filter(slope.copy(), Wn)

        return slope, np.linspace(0, self.t, data.shape[0])

    def initdata(self, posedata):
        posestart = posedata[354:, :]
        # posestart = posedata
        posestart[:, 2] = 0.1172 * np.ones(posestart.shape[0])
        for i in range(posestart.shape[0]):
            posestart[i,3:6] = p.getEulerFromQuaternion(posestart[i,3:7])
        puck_posori = posestart[:,:6]
        puck_posori[:,3:5] = np.zeros(puck_posori[:,3:5].shape)

        puck_posori2 = np.concatenate((puck_posori[:, :2], puck_posori[:, 5:]), axis=1)
        data_ = np.zeros((puck_posori2.shape[0], 2))

        data_[:, 0] = np.linalg.norm(puck_posori2[:, :2], axis=1)
        data_[:, 1] = puck_posori2[:, 2]   # position already exist noise

        return puck_posori, data_

    def runbag(self, readidx, posestart):
        while readidx != posestart.shape[0]:
            # p.stepSimulation()
            if readidx == 0:
                lastpuck = posestart[readidx, 0:3]

            p.resetBasePositionAndOrientation(self.puck, posestart[readidx, 0:3], posestart[readidx, 3:7])
            p.addUserDebugLine(lastpuck, posestart[readidx, 0:3], lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
            lastpuck = posestart[readidx, 0:3]
            readidx += 1

    def get_sim(self, connection_mode=None):
        bag = rosbag.Bag('./rosbag_data/2020-12-04-12-41-02.bag')
        puck_poses, _, _, self.t = self.read_bag(bag)
        # if connection_mode is None:
        #     self._client = p.connect(p.SHARED_MEMORY)
        #     if self._client >= 0:
        #         None
        #     else:
        #         connection_mode = p.DIRECT
        # self._client = p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
        p.connect(p.DIRECT)  # use build-in graphical user interface, p.DIRECT: pass the final results
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0., 0., -9.81)
        p.setTimeStep(1 / 120)

        ####################################################################################################################
        # p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89,
        #                              cameraTargetPosition=[1.55, 0.85, 1.])

        file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
        self.table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
        file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
        self.puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0])

        # init plot
        pick = [0, 1, 5]
        label = ('x', 'y', 'z', 'wx', 'wy', 'wz')
        color = ('r', 'g', 'b', 'k', 'y', 'g')
        # fig, axes = plt.subplots(len(pick), 1)
        # fig1, axes1 = plt.subplots(len(pick), 1)



        p.changeDynamics(self.table, 0, restitution=self.res0)  # 0.2
        p.changeDynamics(self.table, 1, restitution=self.res1)  # 1.1
        p.changeDynamics(self.table, 2, restitution=self.res2)  # 1.1
        # p.changeDynamics(self.table, 3, restitution=self.res3)  # 1
        p.changeDynamics(self.table, 4, restitution=self.res4)  # 1
        p.changeDynamics(self.table, 5, restitution=self.res5)  # 0.8
        # p.changeDynamics(self.table, 6, restitution=self.res6)  # 0.8
        p.changeDynamics(self.table, 7, restitution=self.res7)  # 0.8
        p.changeDynamics(self.table, 8, restitution=self.res8)  # 0.8

        p.changeDynamics(self.table, 0, lateralFriction=self.latf)  # 0.01
        p.changeDynamics(self.puck, 0, restitution=1)  # 1

        puck_posori_6, puck_posori_2 = self.initdata(puck_poses)
        speed, t_series = self.Diff(puck_posori_2, 10, 0.4)

        posestart = puck_poses[354:, :]
        tan_theta = (posestart[25, :2] - posestart[0, :2])[1] / (posestart[25, :2] - posestart[0, :2])[0]
        cos_theta = 1 / np.sqrt(np.square(tan_theta) + 1)
        sin_theta = tan_theta / np.sqrt(np.square(tan_theta) + 1)
        initori = [cos_theta, sin_theta]

        linvel = np.zeros((2,))
        for i, s in enumerate(speed[:, 0]):
            if s > 2.9:
                linvel = speed[i, 0]
                angvel = speed[i, 1]
                break
        angvel = self.angvel
        init_linvel = np.hstack((linvel * np.array(initori), 0))
        init_angvel = np.hstack(([0, 0], angvel))

        poses_pos = []
        poses_ang = []
        simvel = []
        readidx = 0
        # self.runbag(readidx, posestart)
        # p.setRealTimeSimulation(1)
        # p.setPhysicsEngineParameter(fixedTimeStep=t_series[-1]/len(t_series))
        p.resetBasePositionAndOrientation(self.puck, posestart[readidx, 0:3], posestart[readidx, 3:7])
        p.resetBaseVelocity(self.puck, linearVelocity=init_linvel, angularVelocity=init_angvel)
        while readidx < speed.shape[0]:
            p.stepSimulation()

            self.collision_filter(self.puck, self.table)
            if readidx == 0:
                lastpuck = posestart[readidx, 0:3]
                poses_pos.append(lastpuck)
                poses_ang.append(posestart[readidx, 3:7])
                readidx += 1

            simvel.append(p.getBaseVelocity(self.puck)[0] + p.getBaseVelocity(self.puck)[1])
            recordpos, recordang = p.getBasePositionAndOrientation(self.puck)
            poses_pos.append(recordpos)
            poses_ang.append(recordang)

            ####################################################################################################################
            # p.addUserDebugLine(lastpuck, recordpos, lineColorRGB=[0.1, 0.1, 0.5], lineWidth=5)

            lastpuck = recordpos
            readidx += 1

        # Nachbearbeitung
        poses_ang = np.array(poses_ang)
        for i in range(poses_ang.shape[0]):
            poses_ang[i, :3] = p.getEulerFromQuaternion(poses_ang[i, :])
        poses_ang = poses_ang[:, :3]

        p.disconnect()
        return t_series, np.hstack((np.array(poses_pos), np.array(poses_ang))), puck_posori_6


    def get_Err(self):
        pick = [0, 1]
        t_series, sim, bag = self.get_sim()
        # Err_x = []
        # Err_y = []
        # Err = [Err_x, Err_y]
        Err = []

        T = np.linspace(0,sim.shape[0], sim.shape[0])
        reward = [np.exp(-0.005 * t) for t in T]

        # for i, p in enumerate(pick):
        #     Err[i].append(
        #         (
        #         np.sqrt(np.square(sim[:, p] - bag[:, p]) * reward)
        #         ).sum()
        #     )
        h = 10
        idxs = []
        idx = 0
        for i, p in enumerate(bag):
            if i+h > sim.shape[0] or idx+h > sim.shape[0]:
                break
            errs = np.linalg.norm(
                    (bag[i, :2]* np.ones((h, 2)) - sim[idx:idx+h, :2]), axis=1
                )
            err = np.min(errs) * np.exp(-0.005 * T[i])
            Err.append(err)
            idx = np.argmin(errs, axis=0) + i
            idxs.append(idx)
        return np.sum(Err), Err




if __name__ == "__main__" :
    # restitution and friction
    #       t_down_rim_l (1),
    #       t_down_rim_r (2),
    #       t_left_rim (4),
    #       t_right_rim (5),
    #       t_up_rim_l (6),
    #       t_up_rim_r (7)
    #       res0
    #       res12
    #       res45
    #       res67
    #       latf
    res0 = 0.2  # 0.2
    res1 = 1  # 1
    res2 = 1.55  # 1.55
    res4 = 1  # 1
    res5 = 1  # 1
    res7 = 0.83  # 0.83
    res8 = 1.2  # 1.2
    latf = 0.9  # 0.9
    x = [res0, res1, res2, res4, res5, res7, res8, latf]
    model = Model(x)
    pick = [0, 1]
    label = ('x', 'y', 'z', 'wx', 'wy', 'wz')
    color = ('r', 'g', 'b', 'k', 'y', 'g')

    # T = np.linspace(0,1342,1342)

    # y = [np.exp(-0.005 *t) for t in T]
    # plt.plot(T, y)
    # plt.show()
    ############################################### get Err of two position #################################
    sum, Err_xy = model.get_Err()
    print('Err of two direction=', sum)


    ############################################### plot only ##############################################
    t_series, sim, bag = model.get_sim()
    fig, axes = plt.subplots(len(pick), 1)
    for i,p in enumerate(pick):
        axes[i].plot(t_series, sim[:, p], label='sim'+'_'+label[p], color=color[p])
        axes[i].plot(t_series, bag[:, p], label='data'+'_'+label[p], color=color[p], alpha=0.2)
    fig.legend()
    plt.show()


