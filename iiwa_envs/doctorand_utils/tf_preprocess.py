import os
import numpy as np
import rospy
import rosbag
import matplotlib.pyplot as plt

def get_start_stamp(bag: rosbag.Bag):
    pos_prev = []
    t_prev = 0
    count = 0
    t_start = 0
    for topic, msg, t in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            pos = np.array([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y, msg.transforms[0].transform.translation.z])
            t_sec = msg.transforms[0].header.stamp.to_sec()
            if t_prev != 0:
                vel = np.linalg.norm((pos - pos_prev) / (t_sec - t_prev))
                if vel > 0.1:
                    if count == 0:
                        t_start = t
                    count += 1
                    if count > 10:
                        return t_start
                else:
                    count = 0
            pos_prev = pos
            t_prev = t_sec

def write_bag(bag_origin, bag_write: rosbag.Bag, t_start):
    count = 0
    for topic, msg, t in bag_origin.read_messages("/tf", t_start):
        time_stamp = rospy.Time(0) + (t - t_start)
        msg.transforms[0].header.stamp = time_stamp
        bag_write.write(topic, msg, time_stamp)
        count += 1
    bag_write.close()

def plot(bag):
    pos = list()
    t = list()
    for topic, msg, stamp_t in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            pos.append([msg.transforms[0].transform.translation.x,
                        msg.transforms[0].transform.translation.y,
                        msg.transforms[0].transform.translation.z])
            t.append(msg.transforms[0].header.stamp.to_sec())

    pos = np.array(pos)
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(t, pos[:, 0])
    axes[1].plot(t, pos[:, 1])
    axes[2].plot(t, pos[:, 2])
    plt.show()


if __name__ == "__main__":
    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/"
    bag_name = "2020-12-04-12-41-02" + ".bag"

    bag = rosbag.Bag(os.path.join(bag_dir, bag_name))
    if not os.path.exists(bag_dir + "cut/"):
        os.mkdir(bag_dir + "cut/")
    bag_write = rosbag.Bag(os.path.join(bag_dir + "cut/", bag_name), 'w')

    t_start = get_start_stamp(bag)
    print(t_start.to_sec() - bag.get_start_time())

    write_bag(bag, bag_write, t_start)

    bag_write = rosbag.Bag(os.path.join(bag_dir + "cut/", bag_name))
    plot(bag_write)
