import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import matplotlib.pyplot as plt
from scipy import interpolate
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
                                   msg.transforms[0].transform.rotation.w,
                                   msg.transforms[0].transform.rotation.x,
                                   msg.transforms[0].transform.rotation.y,
                                   msg.transforms[0].transform.rotation.z,
                                   t_i])
                if len(mallet_poses) == 0 or not np.equal(np.linalg.norm(mallet_poses[-1][:2] - pose_i[:2]), 0):
                    mallet_poses.append(pose_i)

            elif msg.transforms[0].child_frame_id == "Puck":
                count_puck += 1
                pose_i = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   msg.transforms[0].transform.translation.z,
                                   msg.transforms[0].transform.rotation.w,
                                   msg.transforms[0].transform.rotation.x,
                                   msg.transforms[0].transform.rotation.y,
                                   msg.transforms[0].transform.rotation.z,
                                   t_i])
                if len(puck_poses) == 0 or not np.equal(np.linalg.norm(puck_poses[-1][:2] - pose_i[:2]), 0):
                    puck_poses.append(pose_i)

            elif msg.transforms[0].child_frame_id == "Table":
                count_table += 1
                pose_i = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   msg.transforms[0].transform.translation.z,
                                   msg.transforms[0].transform.rotation.w,
                                   msg.transforms[0].transform.rotation.x,
                                   msg.transforms[0].transform.rotation.y,
                                   msg.transforms[0].transform.rotation.z,
                                   t_i])
                if len(table_poses) == 0 or not np.equal(np.linalg.norm(table_poses[-1][2] - pose_i[2]), 0):
                    table_poses.append(pose_i)
    print("Found puck TF: {}, used: {}.".format(count_puck, len(puck_poses)))
    print("Found mallet TF: {}, used: {}.".format(count_mallet, len(mallet_poses)))
    print("Found table TF: {}, used: {}.".format(count_table, len(table_poses)))

    mallet_poses = np.array(mallet_poses)
    puck_poses = np.array(puck_poses)
    table_poses = np.array(table_poses)

    return mallet_poses, puck_poses, table_poses, t_end - t_start

def plot_trajectory(puck_pose, t, table_pose=None, idx=None):
    # plot separate axis
    fig, axes = plt.subplots(3)
    axes[0].set_title("X - Direction")
    axes[1].set_title("Y - Direction")
    axes[2].set_title("Z - Direction")
    if not idx is None:
        for i in range(len(idx) - 1):
            axes[0].plot(puck_pose[idx[i]:idx[i + 1] + 1, -1], puck_pose[idx[i]:idx[i + 1] + 1, 0])
            axes[0].scatter(puck_pose[idx[i], -1], puck_pose[idx[i], 0], s=20)
            # axes[0].text(puck_pose[idx[i], -1], puck_pose[idx[i], 0], str(idx[i]))

            axes[1].plot(puck_pose[idx[i]:idx[i + 1] + 1, -1], puck_pose[idx[i]:idx[i + 1] + 1, 1])
            axes[1].scatter(puck_pose[idx[i], -1], puck_pose[idx[i], 1], s=20)
            # axes[1].text(puck_pose[idx[i], -1], puck_pose[idx[i], 1], str(idx[i]))

            axes[2].plot(puck_pose[idx[i]:idx[i + 1] + 1, -1], puck_pose[idx[i]:idx[i + 1] + 1, 2])
            axes[2].scatter(puck_pose[idx[i], -1], puck_pose[idx[i], 2], s=20)
            # axes[2].text(puck_pose[idx[i], -1], puck_pose[idx[i], 2], str(idx[i]))
    else:
        axes[0].plot(puck_pose[:, -1], puck_pose[:, 0])
        axes[1].plot(puck_pose[:, -1], puck_pose[:, 1])
        axes[2].plot(puck_pose[:, -1], puck_pose[:, 2])

    # axes[0].text(puck_pose[idx[-1], -1], puck_pose[idx[-1], 0], str(idx[-1]))
    # axes[1].text(puck_pose[idx[-1], -1], puck_pose[idx[-1], 1], str(idx[-1]))
    # axes[2].text(puck_pose[idx[-1], -1], puck_pose[idx[-1], 2], str(idx[-1]))

    # plot scatter in x-y plane
    fig, axes = plt.subplots()
    axes.scatter(puck_pose[:, 0], puck_pose[:, 1], marker=".", s=10)
    axes.set_title("X - Y plane")
    if not table_pose is None:
        plt.scatter(table_pose[:, 0], table_pose[:, 1], color='r', marker='x')

    # plot cutted trajectory
    fig, axes = plt.subplots()
    for i in range(len(idx) - 1):
        axes.plot(puck_pose[idx[i]: idx[i + 1] + 1, 0], puck_pose[idx[i]: idx[i + 1] + 1, 1])
        axes.scatter(puck_pose[idx[i], 0], puck_pose[idx[i], 1], marker=".", s=20)
        # axes.text(puck_pose[idx[i], 0], puck_pose[idx[i], 1], str(idx[i]))
    # axes.text(puck_pose[idx[-1], 0], puck_pose[idx[-1], 1], str(idx[-1]))
    axes.set_xlim([0.55, 2.70])
    axes.set_ylim([0.35, 1.40])
    axes.set_aspect(1.0)
    plt.draw()

def plot_velocity(puck_pose):
    diff = puck_pose[1:, :2] - puck_pose[:-1, :2]
    diff_t = puck_pose[1:, -1] - puck_pose[:-1, -1]
    fig, axes = plt.subplots(3)
    axes[0].plot(diff[:, 0] / diff_t)
    axes[0].plot(np.zeros_like(diff[:, 0]), 'r-.')
    axes[0].set_title('X')
    axes[1].plot(diff[:, 1] / diff_t)
    axes[1].plot(np.zeros_like(diff[:, 0]), 'r-.')
    axes[1].set_title('y')
    axes[2].plot(np.linalg.norm(diff[:, :2], axis=-1) / diff_t)
    axes[2].plot(np.zeros_like(diff[:, 0]), 'r-.')
    axes[2].set_title('Magnitude')
    plt.draw()

def getvel(bag):

    mallet_poses, puck_poses, table_poses, t = read_bag(bag)
    t_series = np.linspace(0, t, puck_poses.shape[0])
    t_int_series = np.linspace(0, t, 10000)
    puck_itpl= np.zeros((len(t_int_series), puck_poses.shape[1]))
    puck_itpl_ = np.zeros((len(t_int_series), puck_poses.shape[1]))
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


    return mallet_poses, puck_poses, table_poses, t, slope


if __name__ == "__main__":
    bag = rosbag.Bag('2020-11-20-12-35-23.bag')
    mallet_poses, puck_poses, table_poses, t, slope = getvel(bag)
    t_int_series = np.linspace(0, t, 10000)
    t_series = np.linspace(0, t, puck_poses.shape[0])
    f1 = interpolate.interp1d(t_series, puck_poses[:, 0], kind='slinear')
    f2 = interpolate.interp1d(t_series, puck_poses[:, 0], kind='zero')
    f3 = interpolate.interp1d(t_series, puck_poses[:, 0], kind='nearest')
    f4 = interpolate.interp1d(t_series, puck_poses[:, 0], kind='quadratic')
    y1 = f1(t_int_series)
    y2 = f2(t_int_series)
    y3 = f3(t_int_series)
    y4 = f4(t_int_series)
    fig, ax = plt.subplots(3, 2, sharex='col', sharey='row')
    ax[0, 0].plot(t_int_series, y1,  label='slinear')
    ax[0, 1].plot(t_int_series, y2, label='zero')
    ax[1, 0].plot(t_int_series, y3, label='nearest')
    ax[1, 1].plot(t_int_series, y4, label='quadratic')
    for i in range(8):
        plt.plot(t_int_series, slope[:, i])
    # print('max idx axis0', np.argmax(slope[:,0]))
    # print('max idx axis1', np.argmax(slope, axis=1))
    # print(puck_poses)

    plt.show()



