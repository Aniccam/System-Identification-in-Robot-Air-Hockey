import os
import numpy as np
import matplotlib.pyplot as plt
from plot_bag import plotbag
from plot_bag import plotdata
import tf
import rosbag
from angles import shortest_angular_distance

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
            if  ((  (bagdata[i, 1:3] - bagdata[i-h, 1:3])  )*
                    (  (bagdata[i+h, 1:3] - bagdata[i, 1:3])    ) < 0).any():

                delta.append([i,   (bagdata[i, 1:3] - bagdata[i-h, 1:3])  *
                     (bagdata[i+h, 1:3] - bagdata[i, 1:3]) ])
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

def get_segment_dataset(data, t_stamp):
    shortrim_set = []
    longrim_set = []
    for i, t in enumerate(t_stamp):
        if data[t, 1] > 2.58 or data[t, 1] < 0.8:
            print(t, "short")
            if i == len(t_stamp)-1:
                shortrim_set.append(data[t_stamp[i-1]:, :])
            elif i == 0:
                shortrim_set.append(data[:t_stamp[i+1], :])
            else:
                shortrim_set.append(data[t_stamp[i-1]+10: t_stamp[i+1], :])
        elif data[t, 2] > 1.29 or data[t, 2] < 0.77:
            print(t, "longrim")
            if i == len(t_stamp)-1:
                longrim_set.append(data[t_stamp[i-1]:, :])
            elif i == 0:
                longrim_set.append(data[:t_stamp[i+1], :])
            else:
                longrim_set.append(data[t_stamp[i-1]+10:t_stamp[i+1], :])
        else:
            print('at the corner')

    return shortrim_set, longrim_set

def read_bag(bag):
    pos = []
    ori = []
    t = []
    for topic, msg, _ in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            pos.append([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y,
                        msg.transforms[0].transform.translation.z])
            ori.append([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                        msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
            t.append(msg.transforms[0].header.stamp.to_sec())
    ori = np.array(ori)  # original orientation

    # quaternion 2 euler
    for i, quat in enumerate(ori):
        ori[i, :3] = tf.transformations.euler_from_quaternion(quat)
    ori = ori[:, -2]

    # eliminate angle jump
    elimated_ori = np.zeros((ori.shape[0]))  # reduce to 1 dim
    for i, o in enumerate(ori):
        if i == 0:
            elimated_ori[i] = o
        else:
            elimated_ori[i] = shortest_angular_distance(ori[i-1], ori[i]) + elimated_ori[i-1]
    ori = np.array(ori)
    bagdata = np.zeros((len(t), 7))   # [t * 1, pos * 3, ang * 3]
    bagdata[:, 0] = t
    bagdata[:, 1:4] = np.array(pos)
    # bagdata[:, -1] = elimated_ori
    bagdata[:, -1] = ori

    return bagdata

def plt_segment(set):
    if set != []:
        fig1, axes1 = plt.subplots(len(set), 1)
        for i in range(len(set)):
            axes1[i].plot(np.array(set[i])[:,1], np.array(set[i])[:, 2])
            axes1[i].scatter(np.array(set[i])[0,1], np.array(set[i])[0, 2], color='r', s=20)
        fig1.show()
        # if shortrim_set != []:
        #     fig1, axes1 = plt.subplots(len(shortrim_set), 1)
        #     for i in range(len(shortrim_set)):
        #         axes1[i].plot(np.array(shortrim_set[i])[:,1], np.array(shortrim_set[i])[:, 2])
        #         axes1[i].scatter(np.array(shortrim_set[i])[0,1], np.array(shortrim_set[i])[0, 2], color='r', s=5)
        #     fig1.show()
        #
        # if longrim_set != []:
        #     fig2, axes2 = plt.subplots(len(longrim_set), 1)
        #     for i in range(len(longrim_set)):
        #         axes2[i].plot(np.array(longrim_set[i])[:,1], np.array(longrim_set[i])[:, 2])
        #         axes2[i].scatter(np.array(longrim_set[i])[0,1], np.array(longrim_set[i])[0, 2], color='r', s=5)
        #     fig2.show()

def segment2file():
    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()
    writename1 = 0
    writename2 = 0

    for bag_name in dir_list:
        bag = rosbag.Bag(os.path.join(bag_dir, bag_name))

        bagdata = read_bag(bag)
        t_stamp = get_collide_stamp(bagdata)
        # plotdata(bagdata, 'bag', t_stamp)
        shortrim_set, longrim_set = get_segment_dataset(bagdata, t_stamp)
        shortdir = os.path.join(bag_dir + "/shortrim_collision/")
        longdir = os.path.join(bag_dir + "/longrim_collision/")

        for i in range(len(shortrim_set)):
            short_datadir = os.path.join(shortdir + str(writename1) + ".txt")
            # if not os.path.exists(short_datadir):
            #     os.mkdir(short_datadir)
            file = open(short_datadir, 'w')
            for item in shortrim_set[i]:
                file.write("%s\n" % item)
            file.close()
            writename1 += 1

        for i in range(len(longrim_set)):
            long_datadir = os.path.join(longdir + str(writename2) + ".txt")

            # if not os.path.exists(long_datadir):
            #     os.mkdir(long_datadir)

            file = open(long_datadir, 'w')
            for item in longrim_set[i]:
                file.write("%s\n" % item)
            file.close()
            writename2 += 1


if __name__ == "__main__":
    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited/longrim_collision/unusable/"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()



    i = 39



    filename = dir_list[i]
    print(filename)
    data = []

    f = open(bag_dir + filename, 'r')
    for i, item in enumerate(f):
        if i % 2 != 0:
            pass
        else:
            data.append(np.array(np.float64(item.replace("[", " ").replace("\n", " ").split() ) ))
    data = np.array(data)
    plt.plot(data[:, 1], data[:, 2], label=filename)
    plt.legend()
    plt.show()



