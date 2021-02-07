import os
import numpy as np
import matplotlib.pyplot as plt
from plot_bag import plotbag
from plot_bag import plotdata

import rosbag

def get_collide_stamp(bagdata):

    bagdata = np.around(bagdata, 2)
    h = 3
    stamp = []
    delta = []

    for i in range(bagdata.shape[0]):
        if i < 0:
            pass
        elif i >= bagdata.shape[0] - h:
            break
        else:
            if  ((  (bagdata[i, 1:-1] - bagdata[i-h, 1:-1])  )*
                    (  (bagdata[i+h, 1:-1] - bagdata[i, 1:-1])    ) < 0).any():

                delta.append([i,   (bagdata[i, 1:-1] - bagdata[i-h, 1:-1])  *
                     (bagdata[i+h, 1:-1] - bagdata[i, 1:-1]) ])
            else:
                pass

    delta = np.array(delta)

    for i in range(1, delta.shape[0]):
        if delta[i, 0] - delta[i-1, 0] < 10:
            pass
        else:
            stamp.append(delta[i, 0])
    stamp = np.array(stamp)
    return stamp

def read_bag(bag):
    pos = []
    t = []
    for topic, msg, _ in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            pos.append([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y,
                        msg.transforms[0].transform.translation.z])
            t.append(msg.transforms[0].header.stamp.to_sec())

    bagdata = np.zeros((len(t), 4))
    bagdata[:, 0] = t
    bagdata[:, 1:] = np.array(pos)
    return bagdata


if __name__ == "__main__":
    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()
    bag_name = dir_list[7]
    bag = rosbag.Bag(os.path.join(bag_dir, bag_name))
    print(bag_name)



    pos = []
    t = []
    for topic, msg, _ in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            pos.append([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y, msg.transforms[0].transform.translation.z])
            t.append(msg.transforms[0].header.stamp.to_sec() )


    bagdata = np.zeros((len(t), 4))
    bagdata[:, 0] = t
    bagdata[:, 1:] = np.array(pos)
    plotdata(bagdata, 'whole')

    i = 4
    t_stamp = get_collide_stamp(bagdata)

    # print('all_timestamp', t_stamp)
    #
    # print('timestamp', t_stamp[i])

    plotdata(bagdata, 'segment', t_stamp)
    plt.show(block=True)
    # plt.plot(bagdata[:, 1], bagdata[:, 2], color="b")
    # plt.show()
