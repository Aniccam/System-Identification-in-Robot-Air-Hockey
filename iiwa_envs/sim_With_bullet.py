import os
from robots import __file__ as path_robots
import numpy as np
import time
import matplotlib.pyplot as plt
import rosbag
from create_segment import *
from scipy import signal
import pybullet as p
import pybullet_data


class Model:
    def __init__(self, parameters, init_pos, init_vel):
        self.t_lateral_f = parameters[0] # friction
        self.left_rim = parameters[1]   # resititution
        self.right_rim = parameters[1]  # restitution
        self.four_side_rim = parameters[2] # restitution
        self.init_pos = init_pos
        self.lin_vel = init_vel[:3]
        self.ang_vel = init_vel[3:]
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

        file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
        self.table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
        file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
        # self.puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0], flags=p.URDF_USE_IMPLICIT_CYLINDER)
        self.puck = p.loadURDF(file, self.init_pos, [0, 0, 0.0, 1.0])
        j_puck = self.get_joint_map(self.puck)
        j_table = self.get_joint_map(self.table)
        p.changeDynamics(self.puck, -1, mass=0.013)
        p.changeDynamics(self.puck, -1, lateralFriction=1)
        # p.changeDynamics(self.puck, -1, restitution=0.8)



        # load parameters
        p.changeDynamics(self.puck, -1, restitution=0.67)

        p.changeDynamics(self.table, j_table.get(b'base_joint'), lateralFriction=self.t_lateral_f)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_l'), lateralFriction=1)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_r'), lateralFriction=1)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_top'), lateralFriction=1)  # no collision
        p.changeDynamics(self.table, j_table.get(b'base_left_rim'), lateralFriction=0.5)
        p.changeDynamics(self.table, j_table.get(b'base_right_rim'), lateralFriction=1)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_l'), lateralFriction=1)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_r'), lateralFriction=1)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_top'), lateralFriction=1)  # no collision


        p.changeDynamics(self.table, j_table.get(b'base_joint'), restitution=1)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_l'), restitution=self.four_side_rim)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_r'), restitution=self.four_side_rim)
        p.changeDynamics(self.table, j_table.get(b'base_down_rim_top'), restitution=self.four_side_rim) # no collision
        p.changeDynamics(self.table, j_table.get(b'base_left_rim'), restitution=self.left_rim)
        p.changeDynamics(self.table, j_table.get(b'base_right_rim'), restitution=self.right_rim)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_l'), restitution=self.four_side_rim)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_r'), restitution=self.four_side_rim)
        p.changeDynamics(self.table, j_table.get(b'base_up_rim_top'), restitution=self.four_side_rim) # no collision

        # init puck

        self.collision_filter(self.puck, self.table)
        p.resetBasePositionAndOrientation(self.puck, self.init_pos, [0, 0, 0, 1.] )
        p.resetBaseVelocity(self.puck, linearVelocity=self.lin_vel, angularVelocity=self.ang_vel)
        p.setTimeStep(1 / 120.)
        # START
        pre_pos = self.init_pos
        pos = [pre_pos]  # record position
        vel = [self.lin_vel]  # record velocity
        t_sim = [0]  # record time
        t = 0

        while (np.array(p.getBaseVelocity(self.puck)) > .001).any():
        # while True:
            p.stepSimulation()
            time.sleep(1/120.)
            t += 1/120
            t_sim.append(t)
            new_pos = p.getBasePositionAndOrientation(self.puck)[0]
            pos.append(new_pos)
            vel.append(p.getBaseVelocity(self.puck)[0])
            if mode == 'GUI':
                p.addUserDebugLine(pre_pos, new_pos, lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
                pre_pos = new_pos
        return np.array(t_sim)[:, None], np.array(pos), np.array(vel)


def get_lin_vel(bagdata):

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

        t_stueck = bagdata[-1, 0] / bagdata.shape[0]
        data = bagdata[:, 1:]  # data without time series
        slope = Diff(t_stueck, data, 10, Wn=0.4)
        return slope

def vel2initvel(linvel,bagdata):
    tan_theta = (bagdata[20, 1:3] - bagdata[0, 1:3])[1] / (bagdata[20, 1:3] - bagdata[0, 1:3])[0]
    cos_theta = 1 / np.sqrt(np.square(tan_theta) + 1)
    sin_theta = tan_theta / np.sqrt(np.square(tan_theta) + 1)
    initori = np.array([cos_theta, sin_theta])

    # get whole speed
    begin_slice = linvel[:15, :]
    init_speed = np.linalg.norm(np.max(np.abs(begin_slice), axis=0)[:2])   # root(x**2 + y** 2)

    return init_speed * initori

def get_Err(t_sim, sim_pos, bagdata):
        sim_pos = sim_pos
        Err = []
        h = 10
        idxs = []
        idx = 0
        for i, p in enumerate(bagdata[:, 1:-1]):
            if i+h > sim_pos.shape[0] or idx+h > sim_pos.shape[0]:
                break
            errs = np.linalg.norm(
                    (bag[i, :2]* np.ones((h, 2)) - sim_pos[idx:idx+h, :2]), axis=1
                )
            err = np.min(errs) * np.exp(-0.005 * t_sim[i])
            Err.append(err)
            idx = np.argmin(errs, axis=0) + i
            idxs.append(idx)
        return np.sum(Err), Err

def runbag(bagdata):
        p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
        p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.])

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0., 0., -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


        file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
        table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
        file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
        readidx = 0
        lastpuck = np.hstack((bagdata[readidx, 1:3], 0.117))
        puck = p.loadURDF(file, lastpuck, [0, 0, 0.0, 1.0])
        while readidx < bagdata.shape[0]-1:
            p.resetBasePositionAndOrientation(puck, np.hstack((bagdata[readidx+1, 1:3], 0.117)), [0, 0, 0, 1.])
            p.addUserDebugLine(lastpuck, np.hstack((bagdata[readidx+1, 1:3], 0.117)), lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
            lastpuck = np.hstack((bagdata[readidx, 1:3], 0.117))
            readidx += 1
        p.disconnect()
if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()

    # choose bag file
    bag_name = dir_list[7]
    bag = rosbag.Bag(os.path.join(bag_dir, bag_name))
    print(bag_name)
    bagdata = read_bag(bag) # [n,4]


    # get linear velocity

    lin_vel = get_lin_vel(bagdata.copy())  # [n,3]
    init_pos = np.hstack((bagdata.copy()[0, 1:-1], 0.117)) # [3,]
    #  get init vel + vel at z direction
    init_lin_vel = np.hstack((vel2initvel(lin_vel, bagdata.copy()), 0))
    angvel= 0
    init_vel= np.hstack((init_lin_vel,np.array([0, 0, angvel])) )

    parameters1 = [0.0001, 0.95, 0.8]
    parameters2 = [0.0001, 0.95, 0.8]

    model = Model(parameters1, init_pos, init_vel)
    # runbag(bagdata.copy())
    t_sim, sim_pos, _ = model.sim_bullet('GUI') # [n,] [n,3] [n,3]




    #  data processing

    # get time index where collision happens
    t_stamp_bag = get_collide_stamp(bagdata.copy())
    t_stamp_sim = get_collide_stamp(np.hstack((t_sim, sim_pos)))
    print(t_stamp_sim)
    plt.plot(np.array(sim_pos)[:t_stamp_sim[1], 0], np.array(sim_pos)[:t_stamp_sim[1], 1], color="green", label='sim')
    plt.plot(bagdata.copy()[:t_stamp_bag[1], 1], bagdata.copy()[:t_stamp_bag[1], 2], color='b', label='original')
    plt.legend()
    plt.show()
    None
    # get_Err(t_sim, sim_pos, bagdata.copy())


