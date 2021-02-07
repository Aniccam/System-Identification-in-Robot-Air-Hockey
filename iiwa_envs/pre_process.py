import os
import numpy as np
import rosbag
import rospy
import matplotlib.pyplot as plt


def get_start_time(bag):
    t_prev = 0
    pos_prev = []
    count = 0
    t_start = 0

    for topic, msg, t in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            pos = np.array([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y, msg.transforms[0].transform.translation.z])
            t_sec = msg.transforms[0].header.stamp.to_sec()
            if t_prev != 0:
                vel = np.linalg.norm((pos - pos_prev) / (t_sec - t_prev))
                if vel > 0.2:
                    if count == 0:
                        t_start = t
                    count += 1
                    if count > 10:
                        return t_start
                else:
                    count = 0
            pos_prev = pos
            t_prev = t_sec

def write_in_bag(bag_origin, bag_write:rosbag.Bag, t_start):
    count = 0
    for topic, msg, t in bag_origin.read_messages("/tf", t_start):
        time_stamp = rospy.Time(0) + (t - t_start)  # create time instance with rospy.Time
        msg.transforms[0].header.stamp = time_stamp
        bag_write.write(topic, msg, time_stamp)
        count += 1
    bag_write.close()



if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/"
    dir_list = os.listdir(bag_dir)
    dir_list.sort()
    filenames = []
    for filename in dir_list:
        filenames.append(filename)
    if not os.path.exists(bag_dir + "edited"):
        os.mkdir(bag_dir + "edited/")
    for bag_name in filenames:
        if bag_name == 'edited':
            continue



        bag = rosbag.Bag(os.path.join(bag_dir, bag_name))

        write_obj = rosbag.Bag(os.path.join(bag_dir + "edited", bag_name), 'w')

        t_start = get_start_time(bag)
        print(t_start.to_sec() - bag.get_start_time())
        write_in_bag(bag, write_obj, t_start)
        bag_after = rosbag.Bag(os.path.join(bag_dir + "edited", bag_name))


    #     plot(bag, 'before')
    #     plot(bag_after, 'after')
    #     plt.legend()
    # plt.show()
