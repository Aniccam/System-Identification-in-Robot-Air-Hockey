import matplotlib.pyplot as plt
import numpy as np
import sympy
# from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy import optimize
from torch import nn


def wrap_to_pi(rad):
    while rad > np.pi:
        rad -= 2 * np.pi
    while rad < -np.pi:
        rad += 2 * np.pi
    return rad


class Kinematics_Torch():
    def __init__(self):
        self.l1 = torch.tensor(0.42, requires_grad=False)
        self.l2 = torch.tensor(0.4, requires_grad=False)
        self.l3 = torch.tensor(0.151, requires_grad=False)
        self.j_limit = torch.tensor(np.rad2deg(np.array([[-170, 170], [-120, 120], [-120, 120]])), requires_grad=False)
        self.v_limit = torch.tensor(0.1, requires_grad=False)

    def forward_kinematics(self, q):
        x = self.l1 * torch.cos(q[:, 0]) + self.l2 * torch.cos(q[:, 0] + q[:, 1]) + self.l3 * torch.cos(
            q[:, 0] + q[:, 1] + q[:, 2])
        y = self.l1 * torch.sin(q[:, 0]) + self.l2 * torch.sin(q[:, 0] + q[:, 1]) + self.l3 * torch.sin(
            q[:, 0] + q[:, 1] + q[:, 2])

        return torch.stack((x, y), dim=1)

    def jacobian(self, q):
        j11 = -(self.l1 * torch.sin(q[0]) + self.l2 * torch.sin(q[0] + q[1]) + self.l3 * torch.sin(q[0] + q[1] + q[2]))
        j12 = -(self.l2 * torch.sin(q[0] + q[1]) + self.l3 * torch.sin(q[0] + q[1] + q[2]))
        j13 = -(self.l3 * torch.sin(q[0] + q[1] + q[2]))
        j21 = self.l1 * torch.cos(q[0]) + self.l2 * torch.cos(q[0] + q[1]) + self.l3 * torch.cos(q[0] + q[1] + q[2])
        j22 = self.l2 * torch.cos(q[0] + q[1]) + self.l3 * torch.cos(q[0] + q[1] + q[2])
        j23 = self.l3 * torch.cos(q[0] + q[1] + q[2])

        self.jac = torch.Tensor([[j11, j12, j13], [j21, j22, j23]])
        return self.jac


class Kinematics:
    def __init__(self):
        self.l1 = 0.42
        self.l2 = 0.4
        self.l3 = 0.151
        self.joint_limits = np.deg2rad(np.array([[-170, 170], [-120, 120], [-120, 120]]))
        self.v_limit = np.array(0.1)

    def forward_kinematics(self, q):
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]) + self.l3 * np.cos(q[0] + q[1] + q[2])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1]) + self.l3 * np.sin(q[0] + q[1] + q[2])

        return np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    def jacobian(self, q):
        j11 = -(self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1]) + self.l3 * np.sin(q[0] + q[1] + q[2]))
        j12 = -(self.l2 * np.sin(q[0] + q[1]) + self.l3 * np.sin(q[0] + q[1] + q[2]))
        j13 = -(self.l3 * np.sin(q[0] + q[1] + q[2]))
        j21 = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]) + self.l3 * np.cos(q[0] + q[1] + q[2])
        j22 = self.l2 * np.cos(q[0] + q[1]) + self.l3 * np.cos(q[0] + q[1] + q[2])
        j23 = self.l3 * np.cos(q[0] + q[1] + q[2])

        self.jac = np.array([[j11, j12, j13], [j21, j22, j23]])
        return self.jac

    def jac_nullspace(self, q):
        self.jacobian(q)
        null_space = np.array(sympy.Matrix(self.jac).nullspace()[0]).astype(np.float64).flatten()
        if np.linalg.norm(null_space) != 0:
            null_space = null_space / (np.linalg.norm(null_space) + 1e-12)
        else:
            null_space = np.zeros((3, 1))
        return null_space

    def inverse_kinematics_opt(self, x_des, q_cur):
        def ik_fun(q_cur, x_des):
            x_cur = self.forward_kinematics(q_cur)
            return np.linalg.norm(x_cur - x_des)

        return optimize.minimize(ik_fun, q_cur, args=(x_des), bounds=self.joint_limits)

    def analytical_IK(self, x_des, q_cur, q_fix, fixed_joint=2, gc=-1):
        if fixed_joint == 1:
            raise NotImplementedError()
            # x_prime = np.cos(q_fix) * (x_des[0] - self.l1 * np.cos(q_fix)) + np.sin(q_fix) * (
            #             x_des[1] - self.l1 * np.sin(q_fix))
            # y_prime = -np.sin(q_fix) * (x_des[0] - self.l1 * np.cos(q_fix)) + np.cos(q_fix) * (
            #             x_des[1] - self.l1 * np.sin(q_fix))
            # cos_q3 = (x_prime**2 + y_prime**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
            # if cos_q3 <= 1 + 1e-9 and cos_q3>=-1 - 1e-9:
            #
            #     cos_q3 = np.clip(cos_q3, -1, 1)
            #     q3_1 = np.arccos(cos_q3)
            #     q3_2 = -np.arccos(cos_q3)
            #     q2_1 = np.arctan2(y_prime, x_prime) - np.arctan2(self.l3 * np.sin(q3_1), self.l2 + self.l3 * np.cos(q3_1))
            #     q2_2 = np.arctan2(y_prime, x_prime) - np.arctan2(self.l3 * np.sin(q3_2), self.l2 + self.l3 * np.cos(q3_2))
            #     if wrap:
            #         q2_1 = wrap_to_pi(q2_1)
            #         q2_2 = wrap_to_pi(q2_2)
            #
            #     if np.equal(q3_1, q3_2) and np.equal(q2_1, q2_2):
            #         return True, np.array([[q_fix, q2_1, q3_1]])
            #     else:
            #         q = np.zeros((2, len(q_cur)))
            #         q[0] = np.array([q_fix, q2_1, q3_1])
            #         q[1] = np.array([q_fix, q2_2, q3_2])
            #         return True, q
            # else:
            #     return False, q_cur
        elif fixed_joint == 2:

            q1_prime = np.arctan2(self.l2 * np.sin(q_fix), self.l1 + self.l2 * np.cos(q_fix))
            l12 = np.sqrt(self.l1 ** 2 + self.l2 ** 2 - 2 * self.l1 * self.l2 * np.cos(np.pi - q_fix))

            theta_goal = np.arctan2(x_des[1], x_des[0])

            d_goal = np.linalg.norm(x_des)
            if d_goal == 0:
                return False, q_cur
            else:
                cos_theta_goal_offset = (d_goal ** 2 + l12 ** 2 - self.l3 ** 2) / (2 * d_goal * l12)

            # cos_q23 = (x_des[0]**2 + x_des[1]**2 - self.l1**2 - self.l2**2 - self.l3**2
            #            - 2*self.l1 * self.l2 * np.cos(q_fix))/ (2 * self.l1 * self.l3)
            # if abs(cos_q23)>1:
            #     return False, q_cur
            # else:
            #     q23 = np.arccos(cos_q23)
            #     q3_1 = q23 - q_fix
            #     q3_2 = -q23 - q_fix
            #
            #     a_1 = self.l1 + self.l2 * np.cos(q_fix) + self.l3 * np.cos(q_fix + q3_1)
            #     b_1 = self.l2 * np.sin(q_fix) + self.l3 * np.sin(q_fix + q3_1)
            #
            #     a_2 = self.l1 + self.l2 * np.cos(q_fix) + self.l3 * np.cos(q_fix + q3_2)
            #     b_2 = self.l2 * np.sin(q_fix) + self.l3 * np.sin(q_fix + q3_2)
            #
            #     q_phi_1 = np.arctan2(b_1, a_1)
            #     q_phi_2 = np.arctan2(b_2, a_2)
            #
            #     q1_1 = np.arctan2(x_des[0], x_des[1]) - q_phi_1
            #     q1_2 = np.arctan2(x_des[0], x_des[1]) - q_phi_2
            #
            #     if wrap:
            #         pass
            #         # q1_1 = wrap_to_pi(q1_1)
            #         # q1_2 = wrap_to_pi(q1_2)
            #         # q3_1 = wrap_to_pi(q3_1)
            #         # q3_2 = wrap_to_pi(q3_2)
            #
            #     q = np.zeros((2, len(q_cur)))
            #     q[0] = np.array([q1_1, q_fix, q3_1])
            #     q[1] = np.array([q1_2, q_fix, q3_2])
            #
            #     return True, q

            if np.abs(cos_theta_goal_offset) <= 1 + 1e-9:
                cos_theta_goal_offset = np.clip(cos_theta_goal_offset, -1, 1)
                theta_goal_offset_1 = np.arccos(cos_theta_goal_offset)
                theta_goal_offset_2 = - theta_goal_offset_1
                q1_1 = theta_goal + theta_goal_offset_1 - q1_prime
                q1_2 = theta_goal + theta_goal_offset_2 - q1_prime

                x3_1 = x_des[0] - self.l1 * np.cos(q1_1) - self.l2 * np.cos(q_fix + q1_1)
                y3_1 = x_des[1] - self.l1 * np.sin(q1_1) - self.l2 * np.sin(q_fix + q1_1)
                q3_1 = np.arctan2(y3_1, x3_1) - q_fix - q1_1

                x3_2 = x_des[0] - self.l1 * np.cos(q1_2) - self.l2 * np.cos(q_fix + q1_2)
                y3_2 = x_des[1] - self.l1 * np.sin(q1_2) - self.l2 * np.sin(q_fix + q1_2)
                q3_2 = np.arctan2(y3_2, x3_2) - q_fix - q1_2

                q1_1 = wrap_to_pi(q1_1)
                q3_1 = wrap_to_pi(q3_1)
                q1_2 = wrap_to_pi(q1_2)
                q3_2 = wrap_to_pi(q3_2)

                q = list()
                if q1_1 >= self.joint_limits[0, 0] and q1_1 <= self.joint_limits[0, 1] and \
                        q3_1 >= self.joint_limits[2, 0] and q3_1 <= self.joint_limits[2, 1]:
                    q.append([q1_1, q_fix, q3_1])
                if q1_2 >= self.joint_limits[0, 0] and q1_2 <= self.joint_limits[0, 1] and \
                        q3_2 >= self.joint_limits[2, 0] and q3_2 <= self.joint_limits[2, 1]:
                    q.append([q1_2, q_fix, q3_2])
                if q.__len__() == 0:
                    return False, q_cur

                return True, q
            else:
                raise ValueError("Feasible Interval does not have result!")
        elif fixed_joint == 3:
            raise NotImplementedError()
        elif fixed_joint == 0:
            # link 3 Direction
            psi = q_fix

            x_j2 = x_des - self.l3 * np.array([np.cos(psi), np.sin(psi)])
            dist_j2 = np.linalg.norm(x_j2)
            offset_j2 = np.arctan2(x_j2[1], x_j2[0])

            cosq2 = (dist_j2 ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
            if abs(cosq2) > 1 + 1e-6:
                return False, q_cur
            else:
                cosq2 = np.clip(cosq2, -1, 1)
                q2 = gc * np.arccos(cosq2)

                q1 = - np.arctan2(self.l2 * np.sin(q2), self.l2 * np.cos(q2) + self.l1) + offset_j2

                q3 = psi - q1 - q2

                q1 = wrap_to_pi(q1)
                q3 = wrap_to_pi(q3)

                if q1 < self.joint_limits[0, 0] or q1 > self.joint_limits[0, 1] or \
                        q2 < self.joint_limits[1, 0] or q2 > self.joint_limits[1, 1] or \
                        q3 < self.joint_limits[2, 0] or q3 > self.joint_limits[2, 1]:
                    return False, q_cur
                else:
                    return True, np.array([q1, q2, q3])
        else:
            raise NotImplementedError()

    def get_feasible_interval(self, p, fixed_joint, step_size=0.01):
        if fixed_joint == 2:
            dist = np.linalg.norm(p)
            if dist > self.l1 + self.l2 + self.l3 or \
                    dist < np.sqrt(
                self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * np.cos(self.joint_limits[1, 1])) - self.l3:
                return list()
            else:
                des_min = dist - self.l3
                if abs((self.l1 ** 2 + self.l2 ** 2 - des_min ** 2) / (2 * self.l1 * self.l2)) > 1:
                    q2_offset_1 = 0
                else:
                    q2_offset_1 = np.arccos((self.l1 ** 2 + self.l2 ** 2 - des_min ** 2) / (2 * self.l1 * self.l2))
                des_max = dist + self.l3
                if abs((self.l1 ** 2 + self.l2 ** 2 - des_max ** 2) / (2 * self.l1 * self.l2)) > 1:
                    q2_offset_2 = np.pi
                else:
                    q2_offset_2 = np.arccos((self.l1 ** 2 + self.l2 ** 2 - des_max ** 2) / (2 * self.l1 * self.l2))

                q_max = np.minimum(np.pi - q2_offset_1, self.joint_limits[1, 1])
                q_min = np.maximum(np.pi - q2_offset_2, 0)

                interval_prev = np.array([[q_min, q_max],
                                          [-q_max, -q_min]])

                return interval_prev
        elif fixed_joint == 0:
            # joint_test = np.linspace(np.deg2rad(-180), np.deg2rad(180), 1000)
            #
            # q2 = []
            # for psi in joint_test:
            #     res, q_inv = self.analytical_IK(p, [0, 0], psi, fixed_joint=0, gc=-1)
            #     if res:
            #         q2.append(np.hstack((psi, q_inv)))
            # q2 = np.array(q2).reshape((-1, 4))
            # plt.figure()
            # plt.clf()
            # plt.plot(q2[:, 0], q2[:, 1], c='r')
            # plt.plot(joint_test, np.ones_like(joint_test) * self.joint_limits[0, 0], c='r', linestyle='--')
            # plt.plot(joint_test, np.ones_like(joint_test) * self.joint_limits[0, 1], c='r', linestyle='--')
            # plt.plot(q2[:, 0], q2[:, 2], c='g')
            # plt.plot(joint_test, np.ones_like(joint_test)*self.joint_limits[1, 0], c='g', linestyle='--')
            # plt.plot(joint_test, np.ones_like(joint_test)*self.joint_limits[1, 1], c='g', linestyle='--')
            # plt.plot(q2[:, 0], q2[:, 3], c='b')
            # plt.plot(joint_test, np.ones_like(joint_test)*self.joint_limits[2, 0], c='b', linestyle='--')
            # plt.plot(joint_test, np.ones_like(joint_test)*self.joint_limits[2, 1], c='b', linestyle='--')
            # plt.plot(joint_test, np.zeros_like(joint_test), c='r', linestyle='--')
            # # plt.xlim(-np.pi, np.pi)
            # # plt.ylim(-np.pi, np.pi)
            # plt.title('EE Position: [{:4f}, {:4f}]'.format(p[0], p[1]))
            # # plt.draw()
            # # plt.pause(0.1)
            # plt.show()

            u_bound = [np.pi]
            l_bound = [-np.pi]

            q2_bound = [0, self.joint_limits[1, 1]]
            x = p[0]
            y = p[1]
            for q2_i in q2_bound:
                c = (np.linalg.norm(
                    p) ** 2 + self.l3 ** 2 - self.l1 ** 2 - self.l2 ** 2 - 2 * self.l1 * self.l2 * np.cos(q2_i)) \
                    / (2 * self.l3)
                # y * sin(psi) + x * cos(psi) = c
                if x ** 2 + y ** 2 >= c ** 2:
                    d = np.sqrt(y ** 2 + x ** 2 - c ** 2)
                    psi1 = np.arctan2((y * c + x * d), (x * c - y * d))
                    psi2 = np.arctan2((y * c - x * d), (x * c + y * d))

                    dtheta_dpsi1 = 2 * y * self.l3 * np.cos(psi1) - 2 * x * self.l3 * np.sin(psi1)
                    dtheta_dpsi2 = 2 * y * self.l3 * np.cos(psi2) - 2 * x * self.l3 * np.sin(psi2)

                    if q2_i == 0:
                        if dtheta_dpsi1 > 0:
                            l_bound.append(psi1)
                        else:
                            u_bound.append(psi1)
                        if dtheta_dpsi2 > 0:
                            l_bound.append(psi2)
                        else:
                            u_bound.append(psi2)
                    else:
                        if dtheta_dpsi1 < 0:
                            l_bound.append(psi1)
                        else:
                            u_bound.append(psi1)
                        if dtheta_dpsi2 < 0:
                            l_bound.append(psi2)
                        else:
                            u_bound.append(psi2)

            if len(l_bound) is not len(u_bound):
                raise ValueError("Number of upper and lower bound are not the same")
            else:
                num_bound = len(l_bound)

            l_bound_np = np.concatenate((np.array(l_bound)[:, np.newaxis], np.zeros((num_bound, 1))), axis=1)
            u_bound_np = np.concatenate((np.array(u_bound)[:, np.newaxis], np.ones((num_bound, 1))), axis=1)

            bound = np.vstack((l_bound_np, u_bound_np))
            bound = bound[bound[:, 0].argsort()]

            index = 0
            interval_psi = []
            while index < bound.shape[0]:
                if bound[index, 1] == 0:
                    iter_low = bound[index, 0]
                    index += 1
                    if bound[index, 1] == 1:
                        iter_high = bound[index, 0]
                        # interval.extend([[iter_low, 0.], [iter_high, 1.]])
                        interval_psi.append([iter_low, iter_high])
                        index += 1
                    else:
                        continue
                else:
                    index += 1
            interval_psi = np.array(interval_psi)

            # check interval for joint 1 and 3

            intervals_with_joint_limit = []
            for _interval_i in interval_psi:
                has_low_bound = False
                has_high_bound = False
                inter_low = -np.inf
                inter_high = np.inf

                q_fix_ = _interval_i[0]
                while q_fix_ <= _interval_i[1]:
                    res, q = self.analytical_IK(p, [0., 0., 0.], q_fix_, fixed_joint=fixed_joint, gc=-1)
                    if res:
                        if not has_low_bound:
                            inter_low = q_fix_
                            inter_high = q_fix_
                            has_low_bound = True
                        else:
                            inter_high = q_fix_
                    else:
                        if has_low_bound:
                            if inter_low != inter_high:
                                intervals_with_joint_limit.append([inter_low, inter_high])
                            has_low_bound = False
                    q_fix_ += step_size

                if has_low_bound:
                    intervals_with_joint_limit.append([inter_low, _interval_i[1]])

            if intervals_with_joint_limit.__len__() != 0:
                return np.unique(intervals_with_joint_limit, axis=0)
            else:
                return np.array(intervals_with_joint_limit)
        else:
            raise NotImplementedError()


def generate_test_sample():
    q = []
    q_1, q_2, q_3 = np.meshgrid(np.linspace(-np.deg2rad(179), np.deg2rad(179), 5),
                                np.linspace(-np.deg2rad(179), np.deg2rad(179), 5),
                                np.linspace(-np.deg2rad(179), np.deg2rad(179), 5))
    q = np.stack((q_1.flatten(), q_2.flatten(), q_3.flatten()), axis=1)
    q_r = np.random.uniform([-np.pi], [np.pi], (100, 3))
    q = np.vstack((q, q_r))
    return q


def test_fk_ik(q):
    kine = Kinematics()
    for q_i in q:
        x = kine.foward_kinematics(q_i)
        res, q = kine.analytical_IK(x, [0., 0., 0.], q_i[0], 1)
        if res:
            min_dist = np.minimum(np.linalg.norm(q_i - q[0]), np.linalg.norm(q_i - q[1]))
            if min_dist > 1e-6:
                print("result: ", q_i, q)
        else:
            print(q_i, " No result!")


def line_manifold(x_start, x_stop, n_points):
    dir = np.asarray(x_stop) - np.asarray(x_start)
    interp_point = np.linspace(0, 1, n_points)
    x = x_start + np.outer(interp_point, dir)
    return x


def plot_null(q, i):
    q = np.asarray(q)
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('q3')

    ax.scatter(q[:, 0, 0], q[:, 0, 1], q[:, 0, 2], color='b', alpha=0.5, s=10)
    ax.scatter(q[:, 1, 0], q[:, 1, 1], q[:, 1, 2], color='b', alpha=0.5, s=10)
