import os
import time

import numpy as np
from PyKDL import Rotation

import angles
import rosbag
import rospy
import tf2_geometry_msgs
import tf2_ros
from gazebo_msgs.srv import *
from std_srvs.srv import Empty

import spawn_model


def get_initial_state(bag: rosbag.Bag):
    i = 0
    j = 0
    tf_table = None
    tf_puck_start = None
    tf_puck_10 = None
    for topic, msg, t in bag.read_messages("/tf"):
        if msg.transforms[0].child_frame_id == "Puck":
            if i == 0:
                tf_puck_start = msg.transforms[0]
            if i == 10:
                tf_puck_10 = msg.transforms[0]
                break
            i += 1

        if msg.transforms[0].child_frame_id == "Table":
            if j == 0:
                tf_table = msg.transforms[0]
            j += 1

    puck_start_T = tf2_geometry_msgs.transform_to_kdl(tf_table).Inverse() * \
                   tf2_geometry_msgs.transform_to_kdl(tf_puck_start)
    puck_10_T = tf2_geometry_msgs.transform_to_kdl(tf_table).Inverse() * \
                tf2_geometry_msgs.transform_to_kdl(tf_puck_10)
    _, _, yaw_start = puck_start_T.M.GetRPY()
    _, _, yaw_10 = puck_10_T.M.GetRPY()

    t_start = tf_puck_start.header.stamp.to_sec()
    t_10 = tf_puck_10.header.stamp.to_sec()

    p_start = np.array([puck_start_T.p.x(), puck_start_T.p.y()])
    p_10 = np.array([puck_10_T.p.x(), puck_10_T.p.y()])
    lin_vel_start = (p_10 - p_start) / (t_10 - t_start)
    ang_vel_start = angles.shortest_angular_distance(yaw_start, yaw_10) / (t_10 - t_start)

    return np.hstack([p_start, yaw_start]), np.hstack([lin_vel_start, ang_vel_start]), tf_table


def set_gazebo_puck_state(position, velocity):
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    puck_state = SetModelStateRequest()
    puck_state.model_state.model_name = 'puck_gazebo'
    puck_state.model_state.pose.position.x = position[0]
    puck_state.model_state.pose.position.y = position[1]
    puck_state.model_state.pose.position.z = 0.0
    rot = Rotation.RPY(0, 0, position[2])
    (puck_state.model_state.pose.orientation.x, puck_state.model_state.pose.orientation.y,
     puck_state.model_state.pose.orientation.z, puck_state.model_state.pose.orientation.w) = rot.GetQuaternion()
    puck_state.model_state.twist.linear.x = velocity[0]
    puck_state.model_state.twist.linear.y = velocity[1]
    puck_state.model_state.twist.linear.z = 0.
    puck_state.model_state.twist.angular.z = velocity[2]

    puck_state.model_state.reference_frame = 'air_hockey_table::Table'
    response = set_model_state(puck_state)


def set_gazebo_table_state(tfData):
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    table_state = SetModelStateRequest()
    table_state.model_state.model_name = 'air_hockey_table'
    table_state.model_state.pose.position.x = tfData.transform.translation.x
    table_state.model_state.pose.position.y = tfData.transform.translation.y
    table_state.model_state.pose.position.z = tfData.transform.translation.z
    table_state.model_state.pose.orientation.x = tfData.transform.rotation.x
    table_state.model_state.pose.orientation.y = tfData.transform.rotation.y
    table_state.model_state.pose.orientation.z = tfData.transform.rotation.z
    table_state.model_state.pose.orientation.w = tfData.transform.rotation.w
    table_state.model_state.reference_frame = 'world'
    response = set_model_state(table_state)


def publish_table_tf(tfData):
    tfData.header.stamp = rospy.Time.now()
    br.sendTransform(tfData)

def validate(bag_dir):
    # pause physics get current time
    unpause_physics()
    rospy.sleep(0.1)
    pause_physics()

    bag = rosbag.Bag(bag_dir)
    init_pos, init_vel, tf_table = get_initial_state(bag)

    set_gazebo_table_state(tf_table)
    time.sleep(0.1)
    set_gazebo_puck_state(init_pos, init_vel)

    publish_table_tf(tf_table)

    t_start = rospy.Time.now()
    unpause_physics()
    for topic, msg, t in bag.read_messages("/tf"):
        while rospy.Time.now() - t_start < t - rospy.Time(0):
            time.sleep(1 / 240.)

        if msg.transforms[0].child_frame_id == "Puck":
            tfPuckData = msg.transforms[0]
            tfPuckData.child_frame_id = "PuckData"
            tfPuckData.header.stamp = t_start + rospy.Duration(msg.transforms[0].header.stamp.to_sec())
            tfPuckData.header.frame_id = "world"
            br.sendTransform(tfPuckData)

    pause_physics()

# def

if __name__ == "__main__":
    rospy.init_node("play_bag_tf")
    time.sleep(1)

    rate = rospy.Rate(120)
    br = tf2_ros.TransformBroadcaster()

    unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

    spawn_model.spawn_puck(restitution=1, lateral_friction=1.0, viscous_friction=0.0)
    spawn_model.spawn_table(restitution=0.68, lateral_friction=0.05)

    # Read data
    data_dir = "/home/puze/air_hockey_record/rosbag_data/29-01-2021/cut"
    file_name = "2021-01-29-12-44-24" + ".bag"
    validate(os.path.join(data_dir, file_name))
