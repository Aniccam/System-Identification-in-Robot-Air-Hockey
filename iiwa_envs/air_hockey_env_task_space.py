import time
import numpy as np

from scipy.spatial.transform import Rotation

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box

from air_hockey.environments.iiwa_envs import AirHockeyBase
from air_hockey.robots.iiwa.kinematics import Kinematics, IIWA_JOINT_MIN_LIMITS, IIWA_JOINT_MAX_LIMITS

try:
    from ias_pykdl.kdl_interface import KDLInterface
except ImportError:
    KDLInterface = None


class AirHockeyTaskSpaceEnv(Environment):
    def __init__(self, gamma=0.995, horizon=1500, debug_gui=False, kine_type='analytical'):
        self.simulation = AirHockeyBase(gamma, horizon, debug_gui)
        self.kin_type = kine_type

        # [X, Y, Pitch, Yaw, Psi]
        low = np.array([0.6, -0.5]) #, np.deg2rad(-180), np.deg2rad(-10), 0])
        high = np.array([1.1, 0.5]) #, np.deg2rad(180), np.deg2rad(75), 0])
        action_space = Box(low, high)

        observation_space = self.simulation._mdp_info.observation_space
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

        self.table_surface_height = 0.162
        self.iiwa_global_config = [1, -1, 1]
        self.fixed_parameters = [0., np.deg2rad(-35), 1.7]
        # self.fixed_parameters = [0., np.deg2rad(-35), 0]

        self.get_initial_state()

        if self.kin_type == 'analytical':
            self.kinematics = Kinematics()
        elif self.kin_type == 'pykdl' and KDLInterface is not None:
            self.kinematics = KDLInterface(self.simulation.robot_path, 7, 'front_base', 'F_link_ee',
                                           np.array(IIWA_JOINT_MIN_LIMITS), np.array(IIWA_JOINT_MAX_LIMITS),
                                           np.array([0., 0., -9.81]))
        else:
            raise NotImplementedError

    def get_initial_state(self):
        p = np.array([0.6, 0.0, self.table_surface_height])
        quat = Rotation.from_euler('YZY', [np.pi, self.fixed_parameters[0], self.fixed_parameters[1]],
                                  degrees=False).as_quat()
        psi = self.fixed_parameters[2]

        kinematics_tmp = Kinematics(tcp_pos=np.array([0., 0., 0.16]))

        pose = np.concatenate((p, quat))
        res, action_out = kinematics_tmp.inverse_kinematics(pose, psi, self.iiwa_global_config)
        init_state = np.concatenate((action_out, [self.fixed_parameters[1]]))
        self.simulation.iiwa_initial_state = init_state

    def reset(self, state=None):
        self._state = self.simulation.reset()
        self.last_action = self.simulation.iiwa_initial_state
        self.is_hitted = False
        return self._state

    def step(self, action):
        cur_obs = self._state
        puck_vel = np.sqrt(cur_obs[7]**2 + cur_obs[8]**2 + cur_obs[9]**2)
        if not self.is_hitted and puck_vel>0.1:
            self.is_hitted=True

        is_reachable, action_q = self._preprocess_action(action)

        if is_reachable:
            self._state, reward, absorbing, _ = self.simulation.step(action_q)
            self.last_action = action_q
        else:
            self._state, reward, absorbing, _ = self.simulation.step(self.last_action)
        time.sleep(1/240.)
        return self._state, reward, absorbing, {}

    def render(self):
        self.simulation.render()

    def _preprocess_action(self, action):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.

        Returns:
            The action to be used for the current step
        """
        if self.is_hitted:
            res = False
            action_out = None
        else:
            if action.shape[0] == 2:
                action = np.concatenate((action, self.fixed_parameters))

            p = np.array([action[0], action[1], self.table_surface_height])
            rot = Rotation.from_euler('YZY',[np.pi, action[2], action[3]], degrees=False)
            rot_mat = rot.as_matrix()
            p = p - 0.16 * rot_mat[:, 2]
            psi = action[4]
            if self.kin_type == 'analytical':
                res, action_out = self.kinematics.inverse_kinematics(p, rot_mat, psi, self.iiwa_global_config)
            elif self.kin_type == 'pykdl':
                action_out = self.kinematics.inverse_kinematics(p, rot.as_euler('zyx'), self.last_action[:7])
                res = True
            else:
                raise NotImplementedError
            action_out = np.concatenate((action_out, [action[3]]))
        return res, action_out


if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.algorithms import Agent

    class DummyAgent(Agent):
        def __init__(self, action_space):
            self._action_space = action_space
            self.last_action = np.random.uniform(self._action_space.low, self._action_space.high)

        def draw_action(self, state):
            self.last_action += np.random.uniform(self._action_space.low, self._action_space.high) * 0.01
            return self.last_action

        def episode_start(self):
            self.last_action = np.random.uniform(self._action_space.low, self._action_space.high)

        def fit(self, dataset):
            pass

    mdp = AirHockeyTaskSpaceEnv()
    agent = DummyAgent(mdp.info.action_space)

    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=10, render=True)
    print("mdp_info state shape", mdp.info.observation_space.shape)
    print("actual state shape", dataset[0][0].shape)
    print("mdp_info action shape", mdp.info.action_space.shape)
    print("actual action shape", dataset[0][1].shape)
