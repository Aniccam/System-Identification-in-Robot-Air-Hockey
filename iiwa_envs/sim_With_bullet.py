import os
import numpy as np
import time
# import matplotlib.pyplot as plt
import rosbag
from create_segment import *
from scipy import signal
import pybullet as p
import pybullet_data
import math
class Model:
    def __init__(self, parameters, init_pos, init_vel):
        """

        :param parameters: [t_lateral_f, left_right_rim_res, left_rim_f, four_side_rim_res, four_side_rim_latf, angvel]  [5,]
        :param init_pos:
        :param init_vel:
        """
        self.t_lateral_f = 0.01  # friction
        self.left_rim_res = 1  #  resititution
        self.right_rim_res = 1   # restitution
        self.left_rim_f = 1  # friction

        self.four_side_rim_res = parameters[0]  # restitution
        self.four_side_rim_latf = parameters[1]  # friction
        # self.angvel = parameters[5]



        self.init_pos = init_pos
        self.lin_vel = init_vel[:3]
        self.ang_vel = init_vel[3:]
        # self.ang_vel[2] = self.angvel     ############# if tune angular velocity
    def get_joint_map(self, bodyUniqueId):

        """

        :param bodyUniqueId:

        :return: dict, val of joint name, joint idx
        """
        idx_num = p.getNumJoints(bodyUniqueId)
        joint_dict = {}
        for i in range(idx_num):
            jointName = p.getJointInfo(bodyUniqueId, i)[1]  # 2nd idx of return is jointName
            joint_dict.update({jointName : i})

        return joint_dict

    def collision_filter(self, puck, table):
        p.setCollisionFilterPair(puck, table, -1, 0, 1)
        p.setCollisionFilterPair(puck, table, -1, 1, 1)
        p.setCollisionFilterPair(puck, table, -1, 2, 1)
        p.setCollisionFilterPair(puck, table, -1, 3, 1)
        p.setCollisionFilterPair(puck, table, -1, 4, 1)
        p.setCollisionFilterPair(puck, table, -1, 5, 1)
        p.setCollisionFilterPair(puck, table, -1, 6, 1)
        p.setCollisionFilterPair(puck, table, -1, 7, 1)
        p.setCollisionFilterPair(puck, table, -1, 8, 1)

    def sim_bullet(self, mode='GUI'):

        if mode == "GUI":
            self.mode = "GUI"
            p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
            p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.])
        elif mode == "DIRECT":
            p.connect(p.DIRECT)
        else:
            print("Input wrong simulation mode ")

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0., 0., -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        path_robots = "/home/hszlyw/PycharmProjects/ahky/robots"
        # tablesize [2.14 x 1.22]
        file = os.path.join(path_robots, "models", "air_hockey_table", "model.urdf")
        self.table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
        file = os.path.join(path_robots, "models", "puck", "model2.urdf")
        # self.puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0], flags=p.URDF_USE_IMPLICIT_CYLINDER)
        self.puck = p.loadURDF(file, self.init_pos, [0, 0, 0.0, 1.0],flags=p.URDF_USE_INERTIA_FROM_FILE or p.URDF_MERGE_FIXED_LINKS)
        j_puck = self.get_joint_map(self.puck)
        j_table = self.get_joint_map(self.table)
        p.changeDynamics(self.puck, -1, lateralFriction=1)
        p.changeDynamics(self.puck, -1, restitution=1)
        # p.changeDynamics(self.puck, -1, restitution=0.8)

        # load parameters

        p.changeDynamics(self.table, j_table.get(b'base_joint'), lateralFriction=self.t_lateral_f)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_l'), lateralFriction=self.four_side_rim_latf)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_r'), lateralFriction=self.four_side_rim_latf)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_top'), lateralFriction=1)  # no collision
        p.changeDynamics(self.table, j_table.get(b'base_left_rim'), lateralFriction=self.left_rim_f)
        p.changeDynamics(self.table, j_table.get(b'base_right_rim'), lateralFriction=self.left_rim_f)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_l'), lateralFriction=self.four_side_rim_latf)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_r'), lateralFriction=self.four_side_rim_latf)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_top'), lateralFriction=1)  # no collision


        p.changeDynamics(self.table, j_table.get(b'base_joint'), restitution=0.2)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_l'), restitution=self.four_side_rim_res)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_r'), restitution=self.four_side_rim_res)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_top'), restitution=self.four_side_rim_res) # no collision
        p.changeDynamics(self.table, j_table.get(b'base_left_rim'), restitution=self.left_rim_res)
        p.changeDynamics(self.table, j_table.get(b'base_right_rim'), restitution=self.right_rim_res)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_l'), restitution=self.four_side_rim_res)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_r'), restitution=self.four_side_rim_res)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_top'), restitution=self.four_side_rim_res) # no collision

        # init puck

        self.collision_filter(self.puck, self.table)
        p.resetBasePositionAndOrientation(self.puck, self.init_pos, [0, 0, 0, 1.] )
        p.resetBaseVelocity(self.puck, linearVelocity=self.lin_vel, angularVelocity=self.ang_vel)
        # START
        pre_pos = self.init_pos
        pos = [pre_pos]  # record position
        vel = [self.lin_vel]  # record velocity
        t_sim = [0]  # record time
        t = 0
        p.setTimeStep(1 / 120.)
        while (np.abs(np.array(p.getBaseVelocity(self.puck)[0][:-1])) > .5).any() and np.abs(p.getBasePositionAndOrientation(self.puck)[0][2]) <0.3:
        # while True:
            p.stepSimulation()
            # time.sleep(1/120.)
            # t += 1/120
            t_sim.append(t)
            new_pos = p.getBasePositionAndOrientation(self.puck)[0]
            pos.append(new_pos)
            vel.append(p.getBaseVelocity(self.puck)[0])
            if mode == 'GUI':
                p.addUserDebugLine(pre_pos, new_pos, lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
                pre_pos = new_pos
        p.disconnect()
        return np.array(t_sim)[:, None], np.array(pos), np.array(vel)


def get_vel(bagdata):

        def filter(data, Wn=0.1):
            b, a = signal.butter(3, Wn)  # Wn is the frequency when amplitude drop of 3 dB, smaller Wn, smaller frequency pass

            for i in range(data.shape[1]):
                data[:, i] = signal.filtfilt(b, a, data[:, i], method='gust')
            return data

        def Diff(t_stueck, data, h, Wn=0.3):
            slope = np.zeros((data.shape))
            for i in range(data.shape[0]):  # Differenzenquotienten
                if i > data.shape[0] - h - 1:
                    slope[i, :] = np.zeros((1, data.shape[1]))
                    break
                slope[i, :] = (data[i + h, :] - data[i, :]) / (h * t_stueck)
            filtered_slope = filter(slope.copy(), Wn)
            return filtered_slope

        t_stueck = (bagdata[-1, 0] - bagdata[0, 0]) / bagdata.shape[0]
        data = bagdata[:, 1:]  # data without time series
        slope = Diff(t_stueck, data, 6, Wn=0.4)

        return slope

def back_process(slope):
    # epre process, delete if vel <.01
    for i, vel in enumerate(slope):
        if (np.abs(vel[1:3]) < .01).all():
            if i+5 == slope.shape[0]:
                return i+4

            elif (np.abs(slope[i + 5, 1:3]) < .01).all():
                return i

            else:
                continue
def vel2initvel(vel,bagdata):
    tan_theta = (bagdata[20, 1:3] - bagdata[0, 1:3])[1] / (bagdata[20, 1:3] - bagdata[0, 1:3])[0]
    cos_theta = 1 / np.sqrt(np.square(tan_theta) + 1)
    sin_theta = tan_theta / np.sqrt(np.square(tan_theta) + 1)
    initori = np.array([cos_theta, sin_theta])

    # get whole speed
    begin_slice = vel[:10, :]
    lin_init_speed = np.linalg.norm(np.max(np.abs(begin_slice), axis=0)[:2])   # root(x**2 + y** 2)
    ang_init_vel = np.max(np.abs(vel[:10, 3:]), axis=0)
    return lin_init_speed * initori, ang_init_vel

def get_Err(bagdata, simdata):

        Err = []
        h = 10
        idxs = []
        idx = 0
        for i, p in enumerate(bagdata[:, 1:3]):
            if i+h > simdata.shape[0] or idx+h > simdata.shape[0]:
                break
            errs = np.linalg.norm(
                    (bagdata[i, 1:3]* np.ones((h, 2)) - simdata[idx:idx+h, 1:-1]), axis=1
                )
            err = np.min(errs) * np.exp(-0.005 * simdata[i, 0])
            Err.append(err)
            idx = np.argmin(errs, axis=0) + i
            idxs.append(idx)
        return np.sum(Err)

# def runbag(bagdata):
#         p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
#         p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.])
#
#         p.resetSimulation()
#         p.setPhysicsEngineParameter(numSolverIterations=150)
#         p.setGravity(0., 0., -9.81)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#
#
#         file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
#         table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
#         file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model2.urdf")
#         readidx = 0
#         lastpuck = np.hstack((bagdata[readidx, 1:3], 0.11945))
#         puck = p.loadURDF(file, lastpuck, [0, 0, 0.0, 1.0])
#         while readidx < bagdata.shape[0]-1:
#             p.resetBasePositionAndOrientation(puck, np.hstack((bagdata[readidx+1, 1:3], 0.11945)), [0, 0, 0, 1.])
#             p.addUserDebugLine(lastpuck, np.hstack((bagdata[readidx+1, 1:3], 0.11945)), lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
#             lastpuck = np.hstack((bagdata[readidx, 1:3], 0.11945))
#             readidx += 1
#         p.disconnect()

def Lossfun(bagdata, simdata, mode='GUI'):
    t_stamp_bag = get_collide_stamp(bagdata)
    t_stamp_sim = get_collide_stamp(simdata)
    collide_num = np.min([len(t_stamp_sim), len(t_stamp_bag)])
    # if len(t_stamp_sim)<2 :
    #     return 100
    # else:
    #     front_idx = t_stamp_bag[1]  # time earlier point
    #     back_idx = t_stamp_bag[1] - 25
    #     ang_res_bag = np.arctan( (bagdata[front_idx, 1] - bagdata[back_idx, 1]) / (
    #                 bagdata[front_idx, 2] - bagdata[back_idx, 2]) )
    #
    #     front_idx = t_stamp_sim[1]  # time earlier point
    #     back_idx = t_stamp_sim[1] - 25
    #     ang_res_sim = np.arctan( (simdata[front_idx, 1] - simdata[back_idx, 1]) / (
    #                 simdata[front_idx, 2] - simdata[back_idx, 2]) )
    #
    #     err = np.linalg.norm(ang_res_bag - ang_res_sim) * 180 /math.pi
    if mode == 'GUI':
        plotdata(simdata, 'sim', markers=t_stamp_sim)
        plotdata(bagdata, 'bag', markers=t_stamp_bag)
    else:
        pass
    Loss = 0
    ERR = 0
    ang_errs = []
    l_errs = []
    for i in range(collide_num-1):

        front_idx_bag = t_stamp_bag[i+1]  # time earlier point
        back_idx_bag = t_stamp_bag[i+1] - 15
        front_idx_sim = t_stamp_sim[i+1]  # time earlier point
        back_idx_sim = t_stamp_sim[i+1] - 15
        ang_list = np.arctan2([
            bagdata[front_idx_bag, 2]-bagdata[back_idx_bag, 2], simdata[front_idx_sim, 2]-simdata[back_idx_sim, 2]
                         ],
            [
                bagdata[front_idx_bag, 1] - bagdata[back_idx_bag, 1], simdata[front_idx_sim, 1] - simdata[back_idx_sim, 1]    ]
        ) * 180 / math.pi
        ang_errs.append( np.abs(ang_list[0] - ang_list[1]) )
        l_errs.append(np.linalg.norm(bagdata[front_idx_bag, 1:3] - simdata[front_idx_sim, 1:3]))


    # Loss *= err * np.exp(-i)


    # print(t_stamp_sim)
    # print(t_stamp_bag)
    # plt.plot(np.array(sim_pos)[:t_stamp_sim[1], 0], np.array(sim_pos)[:t_stamp_sim[1], 1], color="green", label='sim')
    # plt.plot(bagdata.copy()[:t_stamp_bag[1], 1], bagdata.copy()[:t_stamp_bag[1], 2], color='b', label='bag')
    # plt.legend()
    # plt.show()

    return  np.sum(ang_errs)

def Lossfun2(bagdata, simdata, mode='GUI'):


    if mode == 'GUI':
        plotdata(simdata, 'sim')
        plotdata(bagdata, 'bag')
    else:
        pass

    ang_bag = np.arctan2([
        bagdata[5, 2]-bagdata[15, 2], bagdata[-5, 2]-bagdata[-15, 2]
                     ],
        [
            bagdata[5, 1] - bagdata[15, 1], bagdata[-5, 1] - bagdata[-15, 1]  ]
    ) * 180 / math.pi

    ang_sim = np.arctan2([
        simdata[5, 2] - simdata[15, 2], simdata[-5, 2] - simdata[-15, 2]
    ],
        [
            simdata[5, 1] - simdata[15, 1], simdata[-5, 1] - simdata[-15, 1]]
    ) * 180 / math.pi

    delta_bag = np.abs(ang_bag[0] - ang_bag[1])
    delta_sim = np.abs(ang_sim[0] - ang_sim[1])

    loss = np.abs(delta_bag - delta_sim)
    return  loss


# def for_bayes():

if __name__ == "__main__":


    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()

    # choose bag file
    bag_name = dir_list[7]
    bag = rosbag.Bag(os.path.join(bag_dir, bag_name))
    print(bag_name)
    bagdata = read_bag(bag)  # [n,7]
    # t_stamp_bag = get_collide_stamp(bagdata[:, :4].copy())


    # get linear velocity

    lin_ang_vel = get_vel(bagdata.copy())  # return [n,6]
    init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945)) # [3,]
    #  get init vel + vel at z direction
    lin_vel, ang_vel = vel2initvel(lin_ang_vel, bagdata.copy())
    init_vel = np.hstack ((np.hstack((lin_vel, 0)), ang_vel  ))

    # parameters = [0.6291124820709229,0.09892826458795258]
    parameters = [ 0.80021938, 0.50901934]

    model = Model(parameters, init_pos, init_vel)

    #  run bag
    # runbag(bagdata.copy())

    t_sim, sim_pos, _ = model.sim_bullet('GUI')  # [n,] [n,3] [n,3]
    simdata = np.hstack((t_sim, sim_pos))   # [n,4]


    plt.plot(bagdata[:,1], bagdata[:,2], label="bagdata")
    plt.plot(simdata[:,1], simdata[:,2], label="simdata")
    plt.legend()


    #  data processing
    loss = Lossfun(bagdata.copy(), simdata.copy())
    print("loss value in grad = ", loss )

    plt.show()

    # get time index where collision happens

    # calculate Loss: Err of angle






