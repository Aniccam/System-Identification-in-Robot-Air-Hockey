import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import torch
from air_hockey.robots.iiwa.kinematics_torch import KinematicsTorch
from air_hockey.robots import __file__ as path_robots
import sys
import pandas as pd
import matplotlib.pyplot as plt
import rosbag
import quaternion
from scipy.spatial.transform.rotation import Rotation as R
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

def read_bag(bag):
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

    return mallet_poses, puck_poses, table_poses, t_end - t_start


def collision_filter():
    p.setCollisionFilterPair(puck, table, 0, 0, 1)
    p.setCollisionFilterPair(puck, table, 0, 1, 1)
    p.setCollisionFilterPair(puck, table, 0, 2, 1)
    p.setCollisionFilterPair(puck, table, 0, 4, 1)
    p.setCollisionFilterPair(puck, table, 0, 5, 1)

def get_vel(bag):

    _, puck_poses, _, t = read_bag(bag)
    for i in range(puck_poses.shape[0]):
        puck_poses[i,3:6] = p.getEulerFromQuaternion(puck_poses[i, 3:7])

    t_series = np.linspace(0, t, puck_poses.shape[0])
    t_int_series = np.linspace(0, t, 10000)
    puck_itpl= np.zeros((len(t_int_series), puck_poses.shape[1]))
    for i in range(8):
        fpuck = interpolate.interp1d(t_series, puck_poses[:, i], kind='quadratic')
        puck_itpl[:, i] = fpuck(t_int_series)
    slope = np.zeros((10000, 8))
    h= 100
    t_stueck = t / 10000.
    for i in range(puck_itpl.shape[0]):  # Differenzenquotienten
        if i > 9000:
            slope[i,:] = np.zeros((1,8))
            break
        slope[i, :] = (puck_itpl[i + h, :] - puck_itpl[i, :]) / (h * t_stueck )


    return puck_itpl, t, slope

def getorg(puck_poses):
    mint = np.min(puck_poses[:, 0:3], axis=0)
    maxt = np.max(puck_poses[:, 0:3], axis=0)
    dis = maxt - mint
    return dis

if __name__ == "__main__":
    p.connect(p.GUI, 1234) # use build-in graphical user interface, p.DIRECT: pass the final results
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(1 / 240)
    p.setGravity(0., 0., -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.])

    bag = rosbag.Bag('./rosbag_data/2020-12-04-12-41-02.bag')
    puck_poses, t, slope = get_vel(bag)

    # max_x 2.33943247795105
    #  max_y 0.9791051890513136
    file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
    table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])

    file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
    puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0])
    collision_filter()

    kinematics = KinematicsTorch(tcp_pos=torch.tensor([0, 0, 0.3455]))


    # get initial velocity
    readidx = 0
    linearVelocity= []
    idx = np.argmax(slope[:, 0])
    linearVelocity= slope[idx, 0:3]  # x, y, z
    angularVelocity = slope[idx, 3:6]

    # p.changeDynamics(puck, -1, lateralFriction=0.3)
    p.changeDynamics(puck, -1, spinningFriction=0)
    p.changeDynamics(puck, -1, rollingFriction=0)
    p.changeDynamics(puck, -1, restitution=0.8)
    # p.changeDynamics(puck, -1, linearDamping=0.8)
    # p.changeDynamics(puck, -1, angularDamping=0)
    p.changeDynamics(table, 0, lateralFriction=0)

    for linkidx in range(8):
    #   p.changeDynamics(table, linkidx, restitution=0.9)
        p.changeDynamics(table, linkidx, spinningFriction=0)
    #   p.changeDynamics(table, linkidx, rollingFriction=0.2)
        p.changeDynamics(table, linkidx, restitution=0.8)

    recordpos = np.zeros((9000, 7))
    p.resetBaseVelocity(puck, linearVelocity=linearVelocity, angularVelocity=angularVelocity)

    while True:
        if readidx == 9000:
            break
        if readidx == 0:
            lastpuck,_ = p.getBasePositionAndOrientation(puck)
        puck_p, puck_quat = puck_poses[readidx, 0:3], puck_poses[readidx, 3:7]
        # p.resetBaseVelocity(puck, linearVelocity=linearVelocity, angularVelocity=angularVelocity)


        p.stepSimulation()
        collision_filter()
        recordpos[readidx, :3], recordpos[readidx, 3:7] = p.getBasePositionAndOrientation(puck)
        p.addUserDebugLine(lastpuck, recordpos[readidx, :3], lineColorRGB= [0.5,0.5,0.5], lineWidth=5)

        lastpuck = recordpos[readidx, :3]
        # print(p.getBasePositionAndOrientation(puck))
        readidx += 1
        time.sleep(1. / 240)



    # # print('totheend')
    # dis = getorg(puck_poses)
    # plt.figure(figsize=10 * dis[:2])
    # plt.plot(np.arange(recordpos[:, 0].shape[0]), recordpos[:, 0], color='red')
    # # plt.scatter(recordpos[0, 0], recordpos[0, 1], s=5, c='red')
    # plt.plot(np.arange(puck_poses[2160:,0].shape[0]), puck_poses[2160:, 0], color='blue')
    # # plt.scatter(puck_poses[0, 0], puck_poses[0, 1], s=5, c='blue')
    # plt.show()

