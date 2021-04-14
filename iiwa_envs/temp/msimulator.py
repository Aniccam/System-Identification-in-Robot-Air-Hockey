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
    # print("Found puck TF: {}, used: {}.".format(count_puck, len(puck_poses)))
    # print("Found mallet TF: {}, used: {}.".format(count_mallet, len(mallet_poses)))
    # print("Found table TF: {}, used: {}.".format(count_table, len(table_poses)))

    mallet_poses = np.array(mallet_poses)
    puck_poses = np.array(puck_poses)
    table_poses = np.array(table_poses)

    return mallet_poses, puck_poses, table_poses, t_end - t_start
def getvel(bag):

    mallet_poses, puck_poses, table_poses, t = read_bag(bag)
    t_series = np.linspace(0, t, puck_poses.shape[0])
    t_int_series = np.linspace(0, t, 10000)
    puck_itpl= np.zeros((len(t_int_series), puck_poses.shape[1]))
    puck_itpl_ = np.zeros((len(t_int_series), puck_poses.shape[1]))
    for i in range(8):
        fpuck = interpolate.interp1d(t_series, puck_poses[:, i], kind='quadratic')
        puck_itpl[:, i] = fpuck(t_int_series)
    puck_itpl_[:9900, :] = puck_itpl[100:, :]
    puck_itpl_[9900:, :] = np.zeros((100, 8))
    err = puck_itpl - puck_itpl_
    slope = 100 / t * err
    print(slope)

    return mallet_poses, puck_poses, table_poses, t, slope

def collision_filter():
    p.setCollisionFilterPair(puck, table, 0, 0, 0)

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
    _, puck_poses, _, t = read_bag(bag)
    init_joint_state = np.zeros(1)
    gettableshape = getorg(puck_poses)
    print('the xyz limits of table=', gettableshape)

    file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
    table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])

    file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
    puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0])

    collision_filter()

    kinematics = KinematicsTorch(tcp_pos=torch.tensor([0, 0, 0.3455]))

    readidx = 0

    t_series = np.linspace(0, t, puck_poses.shape[0])
    t_interp_series = np.linspace(0, t, puck_poses.shape[0] )
    puck_itpl = np.zeros((puck_poses.shape[0] , puck_poses.shape[1]))
    for i in range(8):
        fpuck = interpolate.interp1d(t_series, puck_poses[:, i], kind='slinear')
        puck_itpl[:, i] = fpuck(t_interp_series)
    ## Adjust adduserdebugline
    puck_itpl[:, 2] = [0.1175 for i in range(puck_itpl.shape[0])]
    while True:
        if readidx == 0:
            lastpuck = puck_itpl[readidx, 0:3]
        p.stepSimulation()

        puck_p, puck_quat = puck_itpl[readidx, 0:3], puck_itpl[readidx, 3:7]

        p.addUserDebugLine(lastpuck, puck_p, lineColorRGB=[0.5, 0.5, 0.5], lineWidth=10)
        lastpuck = puck_p
        p.resetBasePositionAndOrientation(puck, puck_p, puck_quat)

        readidx += 1
        time.sleep(1. / 240)
