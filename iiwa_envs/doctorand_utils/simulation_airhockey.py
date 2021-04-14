import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import torch
from air_hockey.robots.iiwa.kinematics_torch import KinematicsTorch
from air_hockey.robots import __file__ as path_robots

import quaternion
from scipy.spatial.transform.rotation import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

JOINT_ID_F = [3, 4, 5, 6, 7, 8, 9, 12, 13]
JOINT_ID_B = [18, 19, 20, 21, 22, 23, 24, 27, 28]

class DebugManifoldSlider:
    def __init__(self, name):
        self.q_ids = []
        self.q_ids.append(p.addUserDebugParameter(paramName=name + 'X', rangeMin=0.6, rangeMax=1.2, startValue=0.6))
        self.q_ids.append(p.addUserDebugParameter(paramName=name + 'Y', rangeMin=-1.0, rangeMax=1.0, startValue=0))
        self.q_ids.append(
            p.addUserDebugParameter(paramName=name + 'Z_offset', rangeMin=-0.1, rangeMax=0.3, startValue=0))
        self.q_ids.append(p.addUserDebugParameter(paramName=name + 'Roll', rangeMin=90.1, rangeMax=269.9, startValue=180))
        self.q_ids.append(p.addUserDebugParameter(paramName=name + 'Pitch', rangeMin=-90, rangeMax=90, startValue=-0))
        self.q_ids.append(p.addUserDebugParameter(paramName=name + 'Yaw', rangeMin=-180, rangeMax=180, startValue=0.0))
        self.q_ids.append(
            p.addUserDebugParameter(paramName=name + 'Psi', rangeMin=-np.pi, rangeMax=np.pi, startValue=0))

    def read(self):
        out = list()
        for q_id in self.q_ids:
            out.append(p.readUserDebugParameter(q_id))
        return out


def set_joints(robot, q1, q2):
    for i, q_i in enumerate(q1):
        p.setJointMotorControl2(robot, JOINT_ID_F[i], p.POSITION_CONTROL, targetPosition=q_i)#, maxVelocity=p.getJointInfo(robot, JOINT_ID[i])[11])
    for i, q_i in enumerate(q2):
        p.setJointMotorControl2(robot, JOINT_ID_B[i], p.POSITION_CONTROL, targetPosition=q_i)#, maxVelocity=p.getJointInfo(robot, JOINT_ID[i])[11])
    set_striker_joint()

def set_striker_joint():
    def get_striker_joint_position(pose):
        quat_ee = pose[3:]

        q_mallet_joint_1 = np.arctan2(2 * (quat_ee[2]*quat_ee[3] + quat_ee[0]*quat_ee[1]),
                                      (-quat_ee[0]**2 + quat_ee[1]**2 + quat_ee[2]**2 - quat_ee[3]**2))
        q_mallet_joint_2 = np.arcsin(np.clip((quat_ee[0] * quat_ee[2] - quat_ee[1] * quat_ee[3]) * 2, -1, 1))

        if q_mallet_joint_1 > np.pi / 2:
            q_mallet_joint_1 -= np.pi
        elif q_mallet_joint_1 < -np.pi / 2:
            q_mallet_joint_1 += np.pi

        return [q_mallet_joint_1, q_mallet_joint_2]

    for i in range(7):
        q1[i] = p.getJointState(iiwa, JOINT_ID_F[i])[0]
        q2[i] = p.getJointState(iiwa, JOINT_ID_B[i])[0]
    pose_1 = kinematics.forward_kinematics(torch.tensor(q1[:7])).detach().numpy()
    pose_2 = kinematics.forward_kinematics(torch.tensor(q2[:7])).detach().numpy()

    q_striker_1 = get_striker_joint_position(pose_1)
    q_striker_2 = get_striker_joint_position(pose_2)

    p.setJointMotorControl2(iiwa, JOINT_ID_F[-2], p.POSITION_CONTROL, targetPosition=q_striker_1[0])
    p.setJointMotorControl2(iiwa, JOINT_ID_F[-1], p.POSITION_CONTROL, targetPosition=q_striker_1[1])
    p.setJointMotorControl2(iiwa, JOINT_ID_B[-2], p.POSITION_CONTROL, targetPosition=q_striker_2[0])
    p.setJointMotorControl2(iiwa, JOINT_ID_B[-1], p.POSITION_CONTROL, targetPosition=q_striker_2[1])

def draw_frame(position, rotation, replace_ids):
    scale = 0.2
    frame_x = p.addUserDebugLine(position, position + scale * rotation[:, 0], [1., 0., 0.],
                                 replaceItemUniqueId=replace_ids[0])
    frame_y = p.addUserDebugLine(position, position + scale * rotation[:, 1], [0., 1., 0.],
                                 replaceItemUniqueId=replace_ids[1])
    frame_z = p.addUserDebugLine(position, position + scale * rotation[:, 2], [0., 0., 1.],
                                 replaceItemUniqueId=replace_ids[2])
    return [frame_x, frame_y, frame_z]


def get_joint_pos(puck_position, rotation, psi):
    ee_rotation = R.from_euler('zyx', [rotation[2], rotation[1], rotation[0]], degrees=True)
    ee_position = puck_position + np.array([0, 0, 0.1915])
    ee_pose = np.concatenate([ee_position, ee_rotation.as_quat()[3:], ee_rotation.as_quat()[:3]])
    res, q = kinematics.inverse_kinematics(torch.from_numpy(ee_pose), torch.tensor(psi).double(), [1, -1, 1])
    return res, q.detach().numpy()


def collision_filter():
    # disable the collision with left and right rim Because of the inproper collision shape
    for id_f in JOINT_ID_F:
        p.setCollisionFilterPair(iiwa, table, id_f, 0, 0)
        p.setCollisionFilterPair(iiwa, table, id_f, 1, 0)
        p.setCollisionFilterPair(iiwa, table, id_f, 2, 0)
        p.setCollisionFilterPair(iiwa, table, id_f, 4, 0)
        p.setCollisionFilterPair(iiwa, table, id_f, 5, 0)
    for id_b in JOINT_ID_B:
        p.setCollisionFilterPair(iiwa, table, id_b, 0, 0)
        p.setCollisionFilterPair(iiwa, table, id_b, 1, 0)
        p.setCollisionFilterPair(iiwa, table, id_b, 2, 0)
        p.setCollisionFilterPair(iiwa, table, id_b, 4, 0)
        p.setCollisionFilterPair(iiwa, table, id_b, 5, 0)


if __name__ == "__main__":
    p.connect(p.GUI, 1234)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(1 / 240.)
    p.setGravity(0., 0., -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-90.0, cameraPitch=-45, cameraTargetPosition=[0., 0., 1.])
    # planeId = p.loadURDF("plane.urdf")
    iiwa = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "iiwa", "urdf", "air_hockey.urdf")
    iiwa = p.loadURDF(iiwa, basePosition=[0., 0., 0.], baseOrientation=[0., 0., 0., 1.], useFixedBase=True,
                      flags=p.URDF_USE_INERTIA_FROM_FILE)

    init_joint_state = np.zeros(7)

    for i in range(7):
        p.resetJointState(iiwa, JOINT_ID_F[i], init_joint_state[i])
        p.resetJointState(iiwa, JOINT_ID_B[i], init_joint_state[i])

    file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
    table = p.loadURDF(file, [1.73, 0, 0.110], [0, 0, 0.0, 1.0])

    file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
    puck = p.loadURDF(file, [1.73, 0, 0.1125], [0, 0, 0.0, 1.0])

    collision_filter()

    kinematics = KinematicsTorch(tcp_pos=torch.tensor([0, 0, 0.3455]))

    frameId = [0, 0, 0]

    manifold_slider_front = DebugManifoldSlider('F')
    manifold_slider_back = DebugManifoldSlider('B')

    # replace_id_F = [0, 1, 2]
    # replace_id_B = [3, 4, 5]
    while True:
        p.stepSimulation()

        puck_p, puck_quat = p.getBasePositionAndOrientation(puck)

        read_out_front = manifold_slider_front.read()
        manifold_pos_front = read_out_front[:3]
        manifold_rot_front = read_out_front[3:6]
        psi_front = read_out_front[6:]
        res, q1 = get_joint_pos(manifold_pos_front, manifold_rot_front, psi_front)
        if not res:
            q1 = np.zeros(JOINT_ID_F.__len__())
            for i in range(JOINT_ID_F.__len__()):
                q1[i] = p.getJointState(iiwa, JOINT_ID_F[i])[0]

        read_out_back = manifold_slider_back.read()
        manifold_pos_back = read_out_back[:3]
        manifold_rot_back = read_out_back[3:6]
        psi_back = read_out_back[6:]
        res, q2 = get_joint_pos(manifold_pos_back, manifold_rot_back, psi_back)
        if not res:
            q2 = np.zeros(JOINT_ID_F.__len__())
            for i in range(JOINT_ID_B.__len__()):
                q2[i] = p.getJointState(iiwa, JOINT_ID_B[i])[0]

        set_joints(iiwa, q1, q2)

        # pos_F, rot_F = p.getLinkState(iiwa, 12)[:2]
        # pos_B, rot_B = p.getLinkState(iiwa, 27)[:2]
        # replace_id_F = draw_frame(pos_F, R.from_quat(rot_F).as_matrix(), replace_id_F)
        # replace_id_B = draw_frame(pos_B, R.from_quat(rot_B).as_matrix(), replace_id_B)

        time.sleep(1. / 240)
