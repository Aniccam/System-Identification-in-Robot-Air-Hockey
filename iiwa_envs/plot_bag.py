import os
import rosbag
import matplotlib.pyplot as plt
import numpy as np



def plotbag(bag, bagtype):
    pos = []
    t = []
    for topic, msg, _ in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == 'Puck':
            pos.append([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y ,msg.transforms[0].transform.translation.z])
            t.append(msg.transforms[0].header.stamp.to_sec())

    pos = np.array(pos)
    fig, axes = plt.subplots(4,1)
    axes[0].plot(t, pos[:, 0], label='x_' + bagtype, color='b')
    axes[1].plot(t, pos[:, 1], label='y_' + bagtype, color='r')
    axes[2].plot(t, pos[:, 2], label='z_' + bagtype, color='g')
    axes[3].plot(pos[:, 0], pos[:, 1], label='xy_' + bagtype, color='y')
    fig.legend()

def plotdata(bagdata, bagtype, markers=None):

    t = bagdata[:, 0]
    pos = np.array(bagdata[:, 1:])
    fig, axes = plt.subplots(4, 1)
    axes[0].plot(t, pos[:, 0], label='x_' + bagtype, color='b')
    axes[1].plot(t, pos[:, 1], label='y_' + bagtype, color='r')
    axes[2].plot(t, pos[:, 2], label='z_' + bagtype, color='g')
    axes[3].plot(pos[:, 0], pos[:, 1], label='xy_' + bagtype, color='y')
    if markers is not None:
        for i in range(axes.shape[0]-1):
            for m in markers:
                axes[i].scatter(t[m], pos[m, i], color='red', s=10)
                axes[-1].scatter(pos[m, 0], pos[m,1],color='red', s=3)
    fig.legend()

if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/"
    dir_list = os.listdir(bag_dir)
    dir_list.sort()
    bag_name = dir_list[8]

    bag_before = rosbag.Bag(os.path.join(bag_dir, bag_name))
    bag_after = rosbag.Bag(os.path.join(bag_dir + 'edited', bag_name))
    plotbag(bag_before, 'before')
    plotbag(bag_after, 'after')

