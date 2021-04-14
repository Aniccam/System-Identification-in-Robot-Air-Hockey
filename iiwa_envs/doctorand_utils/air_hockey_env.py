import os

import numpy as np
import pybullet
import torch

from air_hockey.robots import __file__ as path_robots
from air_hockey.robots.iiwa.kinematics_torch import KinematicsTorch
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.utils.dataset import compute_J


class AirHockeyBase(PyBullet):
    def __init__(self, gamma=0.99, max_steps=1500, debug_gui=False):
        self.robot_path = os.path.join(os.path.dirname(os.path.abspath(path_robots)),
                                       "iiwa", "urdf", "air_hockey.urdf")
        action_spec = [("F_joint_1", pybullet.POSITION_CONTROL),
                       ("F_joint_2", pybullet.POSITION_CONTROL),
                       ("F_joint_3", pybullet.POSITION_CONTROL),
                       ("F_joint_4", pybullet.POSITION_CONTROL),
                       ("F_joint_5", pybullet.POSITION_CONTROL),
                       ("F_joint_6", pybullet.POSITION_CONTROL),
                       ("F_joint_7", pybullet.POSITION_CONTROL),
                       # ("B_joint_1", pybullet.POSITION_CONTROL),
                       # ("B_joint_2", pybullet.POSITION_CONTROL),
                       # ("B_joint_3", pybullet.POSITION_CONTROL),
                       # ("B_joint_4", pybullet.POSITION_CONTROL),
                       # ("B_joint_5", pybullet.POSITION_CONTROL),
                       # ("B_joint_6", pybullet.POSITION_CONTROL),
                       # ("B_joint_7", pybullet.POSITION_CONTROL),
                       ]

        observation_spec = [("puck", PyBulletObservationType.BODY_POS),
                            ("puck", PyBulletObservationType.BODY_LIN_VEL),
                            ("F_joint_1", PyBulletObservationType.JOINT_POS),
                            ("F_joint_1", PyBulletObservationType.JOINT_VEL),
                            ("F_joint_2", PyBulletObservationType.JOINT_POS),
                            ("F_joint_2", PyBulletObservationType.JOINT_VEL),
                            ("F_joint_3", PyBulletObservationType.JOINT_POS),
                            ("F_joint_3", PyBulletObservationType.JOINT_VEL),
                            ("F_joint_4", PyBulletObservationType.JOINT_POS),
                            ("F_joint_4", PyBulletObservationType.JOINT_VEL),
                            ("F_joint_5", PyBulletObservationType.JOINT_POS),
                            ("F_joint_5", PyBulletObservationType.JOINT_VEL),
                            ("F_joint_6", PyBulletObservationType.JOINT_POS),
                            ("F_joint_6", PyBulletObservationType.JOINT_VEL),
                            ("F_joint_7", PyBulletObservationType.JOINT_POS),
                            ("F_joint_7", PyBulletObservationType.JOINT_VEL),
                            ("F_striker_mallet_tip", PyBulletObservationType.LINK_POS),
                            ("F_striker_mallet_tip", PyBulletObservationType.LINK_LIN_VEL),
                            # ("B_joint_1", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_1", PyBulletObservationType.JOINT_VEL),
                            # ("B_joint_2", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_2", PyBulletObservationType.JOINT_VEL),
                            # ("B_joint_3", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_3", PyBulletObservationType.JOINT_VEL),
                            # ("B_joint_4", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_4", PyBulletObservationType.JOINT_VEL),
                            # ("B_joint_5", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_5", PyBulletObservationType.JOINT_VEL),
                            # ("B_joint_6", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_6", PyBulletObservationType.JOINT_VEL),
                            # ("B_joint_7", PyBulletObservationType.JOINT_POS),
                            # ("B_joint_7", PyBulletObservationType.JOINT_VEL),
                            # ("B_striker_mallet_tip", PyBulletObservationType.LINK_POS),
                            # ("B_striker_mallet_tip", PyBulletObservationType.LINK_LIN_VEL),
                            ]

        files = {
            self.robot_path: dict(flags=pybullet.URDF_USE_IMPLICIT_CYLINDER,
                                  basePosition=[0.0, 0, 0.0],
                                  baseOrientation=[0, 0, 0.0, 1.0])
        }

        super().__init__(files, action_spec, observation_spec, gamma, max_steps, debug_gui=debug_gui,
                         size=(250, 250), distance=4, origin=[1.73, 0., 0.], angles=[0., -30, -30])

        # Compute Observation Indices
        self._compute_observation_indices()

        self.iiwa_initial_state = np.array([-9.2250914e-08,  5.8574259e-01,  5.2708398e-08, -1.0400484e+00,
                                            -3.1588606e-08,  1.5158017e+00,  3.1415927e+00])

        self.feasible_range = np.array([[0.55, 1.0],
                                        [-0.46, 0.46],
                                        [0.05, 0.15]])

        self.kinematics = KinematicsTorch(tcp_pos=torch.tensor([0., 0., 0.3455]))

    def _compute_observation_indices(self):
        self._observation_indices_map = dict()
        index_counter = 0

        for name, obs_type in self._observation_map:
            if obs_type is PyBulletObservationType.BODY_POS:
                self._observation_indices_map[name + '_pos'] = list(range(index_counter, index_counter + 7))
                index_counter+=7
            elif obs_type is PyBulletObservationType.BODY_LIN_VEL:
                self._observation_indices_map[name + '_lin_vel'] = list(range(index_counter, index_counter + 3))
                index_counter += 3
            elif obs_type is PyBulletObservationType.BODY_ANG_VEL:
                self._observation_indices_map[name + '_ang_vel'] = list(range(index_counter, index_counter + 3))
                index_counter += 3
            elif obs_type is PyBulletObservationType.LINK_POS:
                self._observation_indices_map[name + '_pos'] = list(range(index_counter, index_counter + 7))
                index_counter+=7
            elif obs_type is PyBulletObservationType.LINK_LIN_VEL:
                self._observation_indices_map[name + '_lin_vel'] = list(range(index_counter, index_counter + 3))
                index_counter += 3
            elif obs_type is PyBulletObservationType.LINK_ANG_VEL:
                self._observation_indices_map[name + '_ang_vel'] = list(range(index_counter, index_counter + 3))
                index_counter += 3
            elif obs_type is PyBulletObservationType.JOINT_POS:
                self._observation_indices_map[name + '_pos'] = list(range(index_counter, index_counter + 1))
                index_counter += 1
            elif obs_type is PyBulletObservationType.JOINT_VEL:
                self._observation_indices_map[name + '_vel'] = list(range(index_counter, index_counter + 1))
                index_counter += 1
            else:
                raise NotImplementedError

    def get_state(self, state, name):
        return state[self._observation_indices_map[name]]

    def setup(self):
        for i, (model_id, joint_id, _) in enumerate(self._action_data[:7]):
            pybullet.resetJointState(model_id, joint_id, self.iiwa_initial_state[i])

        self._set_universal_joint()

        pybullet.resetBasePositionAndOrientation(self._model_map['table'], [1.73, 0, 0.11],
                                                 [0, 0, 0.0, 1.0])

        puck_position_x = 1.12
        puck_position_y = 0.2
        pybullet.resetBasePositionAndOrientation(self._model_map['puck'], [puck_position_x, puck_position_y, 0.1135],
                                                 [0, 0, 0, 1])
        self.collision_filter()

    def collision_filter(self):
        # disable the collision with left and right rim Because of the improper collision shape
        iiwa_links = ['F_link_1', 'F_link_2', 'F_link_3', 'F_link_4', 'F_link_5',
                      'F_link_6', 'F_link_7', 'F_link_ee', 'F_striker_base', 'F_striker_joint_link',
                      'F_striker_mallet', 'F_striker_mallet_tip',
                      'F_link_1', 'B_link_2', 'B_link_3', 'B_link_4', 'B_link_5',
                      'B_link_6', 'B_link_7', 'B_link_ee', 'B_striker_base', 'B_striker_joint_link',
                      'B_striker_mallet', 'B_striker_mallet_tip']
        table_rims = ['t_down_rim_l', 't_down_rim_r', 't_up_rim_r', 't_up_rim_l',
                      't_left_rim', 't_right_rim', 't_base', 't_up_rim_top', 't_down_rim_top']
        for iiwa_l in iiwa_links:
            for table_r in table_rims:
                pybullet.setCollisionFilterPair(self._link_map[iiwa_l][0], self._link_map[table_r][0],
                                                self._link_map[iiwa_l][1], self._link_map[table_r][1], 0)

        pybullet.setCollisionFilterPair(self._model_map['puck'], self._link_map['t_down_rim_top'][0],
                                        -1, self._link_map['t_down_rim_top'][1], 0)
        pybullet.setCollisionFilterPair(self._model_map['puck'], self._link_map['t_up_rim_top'][0],
                                        -1, self._link_map['t_up_rim_top'][1], 0)

    def reward(self, state, action, next_state):
        reward = -1
        return reward

    def is_absorbing(self, state):
        return False

    def _custom_load_models(self):
        table_file = os.path.join(os.path.dirname(os.path.abspath(path_robots)),
                            "models", "air_hockey_table", "model.urdf")
        table = pybullet.loadURDF(table_file, [1.73, 0, 0.0], [0, 0, 0.0, 1.0])

        puck_file = os.path.join(os.path.dirname(os.path.abspath(path_robots)),
                            "models", "puck", "model.urdf")
        puck = pybullet.loadURDF(puck_file, [1.73, 0, 0.1135], [0, 0, -0.7071068, 0.7071068])

        return dict(table=table, puck=puck)

    def _step_finalize(self):
        pass

    def _simulation_pre_step(self):
        pass

    def _apply_control(self, action):
        i = 0
        self._set_universal_joint()
        for model_id, joint_id, mode in self._action_data:
            u = action[i]
            if mode is pybullet.POSITION_CONTROL:
                # kwargs = dict(targetPosition=u, maxVelocity=pybullet.getJointInfo(model_id, joint_id)[11])
                kwargs = dict(targetPosition=u, positionGain=1.0, velocityGain=1.0)
            elif mode is pybullet.VELOCITY_CONTROL:
                kwargs = dict(targetVelocity=u, maxVelocity=pybullet.getJointInfo(model_id, joint_id)[11])
            elif mode is pybullet.TORQUE_CONTROL:
                kwargs = dict(force=u)
            else:
                raise NotImplementedError

            pybullet.setJointMotorControl2(model_id, joint_id, mode, **kwargs)
            # pybullet.resetJointState(model_id, joint_id, u)
            i += 1


    def _set_universal_joint(self):
        def _get_striker_joint_position(pose):
            quat_ee = pose[3:]

            q_mallet_joint_1 = np.arctan2(2 * (quat_ee[2] * quat_ee[3] + quat_ee[0] * quat_ee[1]),
                                          (-quat_ee[0] ** 2 + quat_ee[1] ** 2 + quat_ee[2] ** 2 - quat_ee[3] ** 2))
            q_mallet_joint_2 = np.arcsin(np.clip((quat_ee[0] * quat_ee[2] - quat_ee[1] * quat_ee[3]) * 2, -1, 1))

            if q_mallet_joint_1 > np.pi / 2:
                q_mallet_joint_1 -= np.pi
            elif q_mallet_joint_1 < -np.pi / 2:
                q_mallet_joint_1 += np.pi

            return [q_mallet_joint_1, q_mallet_joint_2]

        q_1 = np.zeros(7)
        for i in range(7):
            q_1[i] = pybullet.getJointState(*self._joint_map['F_joint_{}'.format(i+1)])[0]

        pose_1 = self.kinematics.forward_kinematics(torch.from_numpy(q_1)).detach().numpy()
        q_striker_1 = _get_striker_joint_position(pose_1)

        pybullet.setJointMotorControl2(*self._joint_map['F_striker_joint_1'], controlMode=pybullet.POSITION_CONTROL,
                                       targetPosition=q_striker_1[0], positionGain=1.0, velocityGain=1.0)
        pybullet.setJointMotorControl2(*self._joint_map['F_striker_joint_2'], controlMode=pybullet.POSITION_CONTROL,
                                       targetPosition=q_striker_1[1], positionGain=1.0, velocityGain=1.0)

        # pybullet.resetJointState(*self._joint_map['F_striker_joint_1'], targetValue=q_striker_1[0])
        # pybullet.resetJointState(*self._joint_map['F_striker_joint_2'], targetValue=q_striker_1[1])

if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.algorithms import Agent

    class DummyAgent(Agent):
        def __init__(self, n_actions):
            self._n_actions = n_actions

        def draw_action(self, state):
            return np.random.randn(self._n_actions) * np.pi

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass

    mdp = AirHockeyBase(debug_gui=True)
    agent = DummyAgent(mdp.info.action_space.shape[0])

    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=10, render=True)
    print('reward: ', compute_J(dataset, mdp.info.gamma))
    print("mdp_info state shape", mdp.info.observation_space.shape)
    print("actual state shape", dataset[0][0].shape)
    print("mdp_info action shape", mdp.info.action_space.shape)
    print("actual action shape", dataset[0][1].shape)
