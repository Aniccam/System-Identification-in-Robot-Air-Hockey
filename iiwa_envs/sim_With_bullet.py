import os
import numpy as np
import time
import sys
# import matplotlib.pyplot as plt
import rosbag
from dataprocessing.create_segment import *
from scipy import signal
import pybullet as p
import pybullet_data
import math
from dataprocessing.angles import *


class Model:
    def __init__(self, parameters, init_pos, init_vel):
        """

        :param parameters: [t_lateral_f, left_right_rim_res, left_rim_f, four_side_rim_res, four_side_rim_latf, angvel]  [5,]
        :param init_pos:
        :param init_vel:
        """
        "0.000000472 and damping 0.000000609"
        "seg_dis_33= 5.236729810590324e-07,0.007131593278764413"
        "seg_dis_66= 9.06695006598454e-05,0.006933054232588352"
        "seg_ang_33 = 0.00029251963132992387, 0.0028112339024133717"
        "seg_ang_66 = 8.175343042630247e-06,3.9434919189378935e-06"
        "seg_ang_99 = 0.0001241783920460382,0.008661260161902341"
        self.t_lateral_f = parameters[4] # 0.000000472
        self.left_rim_res = parameters[0]  #  resititution
        self.right_rim_res = parameters[0]  # restitution
        self.left_rim_f = parameters[1] # friction

        self.four_side_rim_res = parameters[0] # restitution
        self.four_side_rim_latf = parameters[1]  # friction
        self.damp = parameters[5] # 0.000000609
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

    def sim_bullet(self, table, mode='GUI'):

        if mode == "GUI":
            self.mode = "GUI"
            p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
            p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.4])
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
        file1 = os.path.join(path_robots, "models", "air_hockey_table", "model.urdf")
        # self.table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
        self.table = p.loadURDF(file1, list(table[:3]), list(table[3:]))
        file2 = os.path.join(path_robots, "models", "puck", "model2.urdf")
        # self.puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0], flags=p.URDF_USE_IMPLICIT_CYLINDER)
        self.puck = p.loadURDF(file2, self.init_pos, [0, 0, 0.0, 1.0],flags=p.URDF_USE_INERTIA_FROM_FILE or p.URDF_MERGE_FIXED_LINKS)

        j_puck = self.get_joint_map(self.puck)
        j_table = self.get_joint_map(self.table)
        tableshape = p.getCollisionShapeData(self.table, j_table.get(b'base_joint'))
        puckshape = p.getCollisionShapeData(self.puck, -1)

        p.changeDynamics(self.puck, -1, lateralFriction=1)
        p.changeDynamics(self.puck, -1, restitution=1)
        p.changeDynamics(self.puck, -1, linearDamping=self.damp)

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
        while (np.abs(np.array(p.getBaseVelocity(self.puck)[0][:-1])) > .1).any() and .115< np.abs(p.getBasePositionAndOrientation(self.puck)[0][2]) < 1.2:
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
        slope = Diff(t_stueck, data, 3, Wn=0.4)

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
def vel2initvel(vel,bagdata, begin_idx):
    margin = 4
    theta = np.arctan2(bagdata[begin_idx+margin, 2] - bagdata[begin_idx, 2],  bagdata[begin_idx+margin, 1] - bagdata[begin_idx, 1] )
    # cos_theta = 1 / np.sqrt(np.square(tan_theta) + 1) * np.sign(bagdata[begin_idx+margin, 1] - bagdata[begin_idx, 1])
    # sin_theta = tan_theta / np.sqrt(np.square(tan_theta) + 1) * np.sign(bagdata[begin_idx+margin, 2] - bagdata[begin_idx,2])

    initori = np.array([np.cos(theta), np.sin(theta)])

    # get whole speed

    lin_init_speed = np.linalg.norm(np.mean(vel[begin_idx:begin_idx+margin, :2], axis=0) ) # root(x**2 + y** 2)

    angs = bagdata[:, -1]
    ang_ls= np.zeros((len(angs), 1))
    ang_ls[0, 0] = angs[0]
    for i,a in enumerate(angs):
        if i == len(angs)-1:
            break
        ang_ls[i+1, 0] = shortest_angular_distance(angs[i], angs[i+1]) + ang_ls[i]

    ang_init_vel = (ang_ls[begin_idx+margin-1] - ang_ls[begin_idx]) / (bagdata[begin_idx+margin-1, 0] - bagdata[begin_idx, 0])
    return np.hstack(( np.hstack((lin_init_speed * initori, 0)), [0, 0, ang_init_vel] ))
    # for i, ang_vel in enumerate(ang_vels):
    #     if i >= len(ang_vels) - 2:
    #         ang_init_vel = ang_vels[25]
    #         return np.hstack(( np.hstack((lin_init_speed * initori, 0)), [0, 0, ang_init_vel] ))
    #     elif ang_vel * ang_vels[i-1] > 0 and np.linalg.norm(ang_vel - ang_vels[i-1]) < 1 and np.linalg.norm(ang_vel - ang_vels[i+1]) < 1 and ang_vel * ang_vels[i+1] > 0 :
    #         ang_init_vel = ang_vel
    #         return np.hstack(( np.hstack((lin_init_speed * initori, 0)), [0, 0, ang_init_vel] ))
    #     else:
    #         continue
def get_Err(bagdata, simdata):

        Err = []
        h = 10
        idxs = []
        idx = 0
        for i, p in enumerate(bagdata[:, 1:3]):
            if i+h > simdata.shape[0] or idx+h > simdata.shape[0]:
                break
            errs = np.linalg.norm(
                    (bagdata[i, 1:3]* np.ones((h, 2)) - simdata[idx:idx+h, 1:3]), axis=1
                )
            err = np.min(errs) * np.exp(-0.004 * i)
            Err.append(err)
            idx = np.argmin(errs, axis=0) + i
            idxs.append(idx)
        return np.sum(Err) / len(Err)
def runbag(bagdata, table_):
        p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
        p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.])

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0., 0., -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        path_robots = "/home/hszlyw/PycharmProjects/ahky/robots"
        file = os.path.join(path_robots, "models", "air_hockey_table", "model.urdf")
        table = p.loadURDF(file, list(table_[:3]), list(table_[3:]) )
        file = os.path.join(path_robots, "models", "puck", "model2.urdf")
        readidx = 0
        lastpuck = np.hstack((bagdata[readidx, 1:3], 0.11945))
        puck = p.loadURDF(file, lastpuck, [0, 0, 0.0, 1.0])
        while readidx < bagdata.shape[0]-1:
            p.resetBasePositionAndOrientation(puck, np.hstack((bagdata[readidx+1, 1:3], 0.11945)), [0, 0, 0, 1.])
            p.addUserDebugLine(lastpuck, np.hstack((bagdata[readidx+1, 1:3], 0.11945)), lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
            lastpuck = np.hstack((bagdata[readidx, 1:3], 0.11945))
            readidx += 1
        p.disconnect()

def Lossfun2(bagdata, simdata, table, mode='GUI'):


    def get_collide_point(data):

        h = 3
        for i in range(h, data.shape[0]):
            if i + h >= data.shape[0]:
                return 6839, 7362
            elif ((
                    (data[i, 1:3] - data[i-h, 1:3])
                    *
                    (data[i+h, 1:3] - data[i, 1:3])
            ) < 0 ).any():
                idx = 0
                before_collide_idx = i
                after_collide_idx = i + 2*h
                # print("2")
                return before_collide_idx, after_collide_idx
            else:
                # print("3")

                continue


    # if mode == 'GUI':
    #     plotdata(simdata, 'sim')
    #     plotdata(bagdata, 'bag')
    # else:
    #     pass
    table = list(table)
    x_ub = table[0] + 0.98
    x_lb = table[0] - 0.98
    y_ub = table[1] + 0.54
    y_lb = table[1] - 0.54
    margin = 6
    b_b, b_a = get_collide_point(bagdata)

    # print("bb ba",b_b, b_a)
    # print("bagdatashape", bagdata.shape[0])

    while b_b - margin < 0:
        margin -= 1
    while b_a + margin > bagdata.shape[0] - 1:
        margin -= 1
    if margin < 0:
        # print("b_b, b_a problems 2")

        return 180.3334
    else:
        ang_bag = np.arctan2([
            bagdata[b_b - margin, 2]-bagdata[b_b, 2], bagdata[b_a + margin, 2]-bagdata[b_a, 2]
                         ],
            [
                bagdata[b_b - margin, 1] - bagdata[b_b, 1], bagdata[b_a + margin, 1] - bagdata[b_a, 1]  ]
        ) * 180 / math.pi

    s_b, s_a = get_collide_point(simdata)
    if s_b == 6839 and s_a == 7362:
        # print("s_b, s_a problems 1")
        return 180.3334
    else:
        while s_b - margin < 0 :
            margin -= 1
        while s_a + margin > simdata.shape[0]-1:
            margin -= 1
        if margin < 0:
            # print("s_b, s_a problems 2")

            return 180.3334
        else:
            # print("sb-margin", s_b, s_b-margin)
            # print("sa+margin", s_a, s_a+margin)

            ang_sim = np.arctan2([
                simdata[s_b - margin, 2]-simdata[s_b, 2], simdata[s_a + margin, 2]-simdata[s_a, 2]
                             ] ,
                [
                    simdata[s_b - margin, 1] - simdata[s_b, 1], simdata[s_a + margin, 1] - simdata[s_a, 1]  ]
            ) * 180 / math.pi

            delta_bag = np.abs(ang_bag[0] - ang_bag[1])
            delta_sim = np.abs(ang_sim[0] - ang_sim[1])
            # loss_ang = np.log( np.abs(delta_bag - delta_sim) + 1)
            loss_ang = np.abs(delta_bag - delta_sim)


            return loss_ang
            ############################################

            ################   linear loss till the end #############

            # endx_ub =  y_lb < (
            #         (bagdata[b_a + margin, 2] - bagdata[b_a, 2] ) / (bagdata[b_a + margin, 1] - bagdata[b_a, 1] ) * (x_ub - bagdata[b_a, 1]) + bagdata[b_a, 2]
            # ) < y_ub
            #
            # endx_lb = y_lb < (
            #         (bagdata[b_a + margin, 2] - bagdata[b_a, 2] ) / (bagdata[b_a + margin, 1] - bagdata[b_a, 1] ) * (x_lb - bagdata[b_a, 1]) + bagdata[b_a, 2]
            # ) < y_ub
            #
            # endy_ub =  x_lb < (
            #         (y_ub - bagdata[b_a, 2]) * (bagdata[b_a + margin, 1] - bagdata[b_a, 1]) / (bagdata[b_a + margin, 2] - bagdata[b_a, 2]) + bagdata[b_a, 1]
            #
            # ) < x_ub
            #
            #
            # endy_lb =  x_lb < (
            #         (y_lb - bagdata[b_a, 2]) * (bagdata[b_a + margin, 1] - bagdata[b_a, 1]) / (bagdata[b_a + margin, 2] - bagdata[b_a, 2]) + bagdata[b_a, 1]
            #     ) < x_ub
            #
            #
            #
            # # loss_dis = print("belongs to none rim")
            #
            #
            #
            # if (simdata[s_a + margin, 1] - simdata[s_a, 1] > 0 ) and endx_ub:
            #     y3_b = (
            #             (bagdata[b_a + margin, 2] - bagdata[b_a, 2]) / (bagdata[b_a + margin, 1] - bagdata[b_a, 1]) * (
            #                 x_ub - bagdata[b_a, 1]) + bagdata[b_a, 2]
            #     )
            #     y3_s = (
            #             (simdata[s_a + margin, 2] - simdata[s_a, 2]) / (simdata[s_a + margin, 1] - simdata[s_a, 1]) * (
            #                 x_ub - simdata[s_a, 1]) + simdata[s_a, 2]
            #     )
            #     loss_dis = np.abs(y3_b - y3_s)
            #     # print("disslos=", np.log(loss_dis * 1000 + 1))
            #     # print("angloss=", loss_ang)
            #     # print("dissweight=", np.log(loss_dis * 1000 + 1) / (np.log(loss_dis * 1000 + 1) + loss_ang))
            #     # print("*" * 100)
            #
            #     return np.log(loss_dis * 1000 + 1) + 2 * loss_ang
            #
            # elif (simdata[s_a + margin, 1] - simdata[s_a, 1] < 0 ) and endx_lb:
            #     y3_b = (
            #             (bagdata[b_a + margin, 2] - bagdata[b_a, 2]) / (bagdata[b_a + margin, 1] - bagdata[b_a, 1]) * (
            #                 x_lb - bagdata[b_a, 1]) + bagdata[b_a, 2]
            #     )
            #     y3_s = (
            #             (simdata[s_a + margin, 2] - simdata[s_a, 2]) / (simdata[s_a + margin, 1] - simdata[s_a, 1]) * (
            #                 x_lb - simdata[s_a, 1]) + simdata[s_a, 2]
            #     )
            #     loss_dis = np.abs(y3_b - y3_s)
            #     # print("disslos=", np.log(loss_dis * 1000 + 1))
            #     # print("angloss=", loss_ang)
            #     # print("dissweight=", np.log(loss_dis * 1000 + 1) / (np.log(loss_dis * 1000 + 1) + loss_ang))
            #     # print("*" * 100)
            #
            #
            #     return np.log(loss_dis * 1000 + 1) + 2*loss_ang
            #
            # elif (simdata[s_a + margin, 2] - simdata[s_a, 2] > 0 ) and endy_ub:
            #     x3_b = (
            #             (y_ub - bagdata[b_a, 2]) * (bagdata[b_a + margin, 1] - bagdata[b_a, 1]) / (
            #                 bagdata[b_a + margin, 2] - bagdata[b_a, 2]) + bagdata[b_a, 1]
            #
            #     )
            #     x3_s = (
            #             (y_ub - simdata[s_a, 2]) * (simdata[s_a + margin, 1] - simdata[s_a, 1]) / (
            #                 simdata[s_a + margin, 2] - simdata[s_a, 2]) + simdata[s_a, 1]
            #
            #     )
            #     loss_dis = np.abs(x3_b - x3_s)
            #     # print("disslos=", np.log(loss_dis * 1000 + 1))
            #     # print("angloss=", loss_ang)
            #     # print("dissweight=", np.log(loss_dis * 1000 + 1) / (np.log(loss_dis * 1000 + 1) + loss_ang))
            #     # print("*" * 100)
            #
            #     return np.log(loss_dis * 1000 + 1) + 2*loss_ang
            #
            # elif (simdata[s_a + margin, 2] - simdata[s_a, 2] < 0 ) and endy_lb:
            #     x3_b = (
            #             (y_lb - bagdata[b_a, 2]) * (bagdata[b_a + margin, 1] - bagdata[b_a, 1]) / (
            #                 bagdata[b_a + margin, 2] - bagdata[b_a, 2]) + bagdata[b_a, 1]
            #
            #     )
            #     x3_s = (
            #             (y_lb - simdata[s_a, 2]) * (simdata[s_a + margin, 1] - simdata[s_a, 1]) / (
            #                 simdata[s_a + margin, 2] - simdata[s_a, 2]) + simdata[s_a, 1]
            #
            #     )
            #     loss_dis = np.abs(x3_b - x3_s)
            #
            #     # print("disslos=", np.log(loss_dis * 1000 + 1))
            #     # print("angloss=", loss_ang)
            #     # print("dissweight=", np.log(loss_dis * 1000 + 1) / (np.log(loss_dis * 1000 + 1) + loss_ang))
            #     # print("*" * 100)
            #     return np.log(loss_dis * 1000 + 1) + 2*loss_ang
            #
            # else:
            #     # print("bug at estimate end point")
            #     return 180.3334



if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/20210224/ori_all/"
    dir_list = os.listdir(bag_dir)
    dir_list.sort()
    count = 0
    Loss = 0
    for filename in dir_list:
        bag = rosbag.Bag(os.path.join(bag_dir, filename))
        data = []

        # with open(bag_dir + filename, 'r') as f:
        #     for line in f:
        #         data.append(np.array(np.float64(line.replace("[", " ").replace("]", " ").replace(",", " ").split())))
        #     bagdata = np.array(data)

        bagdata, table = read_bag(bag)
        # table = np.array([1.7, 0.85, 0.117, 0., 0., 0., 1.])
        table = np.array(table)[0, :]
        # table = bagdata[0, :]
        table[2] = 0.11945
        # bagdata = bagdata[1:, :]
        bagdata[:, 3] = 0.11945 * np.ones(bagdata.shape[0])


    # get linear velocity


        lin_ang_vel = get_vel(bagdata.copy())  # return [n,6]
        # begin_idx = np.argmax(np.abs(lin_ang_vel[:10, 1]))
        for i, vel in enumerate(lin_ang_vel):
            if ( np.abs(vel[0:2]) > 0.1 ).any() and ( np.abs(lin_ang_vel[i+2, 0:2]) > 0.1 ).any():
                begin_idx = i + 10
                break
            else:
                continue

        init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945)) # [3,]
        #  get init vel + vel at z direction

        ###########
        # plt.plot(bagdata[:,1], bagdata[:,2])
        # plt.scatter(bagdata[0,1], bagdata[0,2], color='r', s=10)
        # plt.scatter(bagdata[begin_idx,1], bagdata[begin_idx,2], color='b', s=10)
        #
        # plt.show()


        # init_vel = lin_ang_vel[begin_idx, :]


        init_vel = vel2initvel(lin_ang_vel, bagdata.copy(), begin_idx) # In [n,7]

        # run bag
        # runbag(bagdata.copy(), table)

        parameters = [0.89999997588924,0.0862029979358403, 0.8999999761581421, 0.0117223169654607771]  # loss is angle
        shortparams= [0.699999988079071, 0.4000000059604645, ]
        parameters_comb = [0.5692870164606006, 0.5728446366428586,0.7815188983391684, 0.7779911832945189, 9.459688616606632e-06, 0.06995505922640383]
        parameters_puredis = [0.5000000333785332, 0.47638183117953, 0.015380650174596779, 0.751316530185423, .001, 0.07833489775657654]
        parameters_ang2dis1 = [0.6588862055555563, 0.5588022042031872,0.7120770515310699,0.48903109637226916, 0.005036661922042332, 0.0055209567542747985]
        parameters_pureang= [0.5004560623165386, 0.5697433172992706,0.9146034553520569,0.7792704235951046,0.006211612598792474, 0.04577464769528256]
        parameters_angtimesdis= [0.5016025459847976, 0.4586099838084054, 0.681699731320961, 0.13918265930921014,0.000000472,  0.000000609]
        params_27bagsmult = [0.5350643232102498, 0.019638493115294142, 0.6041054989098973, 0.07778764783505487, 0.001162435774030039,0.0008092878072303286]
        params_27bagsmult0003ex=[0.5948782061362958, 0.6767296591391992, 0.74488450111527, 0.7573606599450943, 0.014027191373899347, 0.0004118297628308313]
        params_aus1 = [0.7322243090496859, 0.40040546232180796, 0.7182018939691728, 0.3882585465536242, 0.008124363608658314, 0.0009918306660691937]
        params_aus2 = [0.7154416441917419, 0.773118393744553, 0.7803165912628174, 2.195339187958629e-13, 0.001052391016855836, 0.0009999909671023488]
        params_aus_w1 = [0.6601211149315492, 0.3743048875766217, 0.7184786981099218, 0.00024820723403912756, 0.01088050939142704, 0.0007365284206054107]
        param_aus3_dis_long = [ 0.929532964993739, 0.13050455779452633, 0.8377456665039062, 0.12618421915495376, 0.0024844819473709067, 0.0004477310517563276]
        param_aus4_ang_long = [0.7781064712815549, 0.7999377967389442, 0.9300000000991063, 0.05591317113149914, 8.882570546120405e-05, 0.00018898608384461957]
        param_aus3_dis_short = [0.8377456665039062, 0.12618421915495376]
        param_aus4_dis_short = [0.9300000000991063, 0.05591317113149914]
        p_all_dis_30=[0.8811311846664506, 0.4105691544176001, 0.9124743731218027, 0.35900088632217547, 0.000000472, 0.000000609] # best obj 0.1410335615239317
        p_all_dis_60=[0.8505313619029673, 0.4771092354335756, 0.8940491783128354, 0.3356883524725823, 0.000000472, 0.000000609]   # best obj 0.1424837147294241
        p_all_dis_90=[0.8834821676450403, 0.7644752860719204, 0.9499845801393345, 0.002574821936491776, 0.000000472, 0.000000609] # best obj 0.14295354676680136
        p_seg_dis_33=[0.8935860395431519, 0.11888450837805967, 0.8891523055307882, 0.053511844080132945,5.236729810590324e-07,0.007131593278764413] # best obj 0.0522040600476066 0.03431201474734212
        p_seg_dis_66=[0.9315619709374023, 0.14591225630032362, 0.7812911370537645, 0.12230765074491501, 9.06695006598454e-05,0.006933054232588352] # best obj 0.048745806677416195 0.03379604156229075

        p_seg_dis_99=[0.9397830035465228, 0.11278709553712492, 0.7820095362422288, 0.1222818575716191,9.111838291033335e-05, 0.007549535277524568] # best obj 0.04878046485241181
        p_seg_ang_33 = [0.8191250007384172, 0.15159492324515833, 0.9499999064754436, 0.09399662911891937, 0.00029251963132992387, 0.0028112339024133717] # best obj 3.7638519312853633 10.38573457698801
        p_seg_ang_66 = [0.9493355324671046, 0.106116896247992, 0.9499999799984453, 0.077954052871899, 8.175343042630247e-06, 3.9434919189378935e-06] # best obj 3.630643092349588 7.849126801038876
        p_seg_ang_99 = [0.7990131849261057, 0.1842469905670424, 0.9499999875915864, 0.13122795522212982, 0.0001241783920460382, 0.008661260161902341] # best obj 2.939860393350947 8.239427807516991


        model = Model(p_seg_dis_66, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'GUI')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))   # [n,4]

        plt.plot(bagdata[:,1], bagdata[:,2], label="bagdata", marker= '.' )
        plt.plot(simdata[:,1], simdata[:,2], label="simdata", marker= '.')
        plt.title(filename)

        plt.axis("equal")
        plt.legend()

        # plt.savefig("/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/traj/"+ filename+'.png', dpi=300)
        loss = get_Err(bagdata.copy(), simdata.copy())
        loss2 = Lossfun2(bagdata.copy(), simdata.copy(), table, 'DIRECT')
        print("number", count, '\n', "lossdis", loss, '\n', "lossang", loss2)
        #  data processing
        # loss = Lossfun2(bagdata.copy(), simdata.copy(), table)
        # print("loss value in grad = ", loss )
        # Loss = Loss + loss
        # # plt.show()
        # print("Loss value in grad = ", Loss )
        # count += 1
        # # get time index where collision happens
        # calculate Loss: Err of angle
        import imageio

        im = imageio.imread('imageio:astronaut.png')
        plt.show()
        None



