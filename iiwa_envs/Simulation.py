import pybullet as p
import numpy as np
import pybullet_data
import time
import os
import rosbag
from robots import __file__ as path_robots
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
def read_bag(bag):
    mallet_poses = []
    puck_poses = []
    table_poses = []
    count_table = 0
    count_puck = 0
    count_mallet = 0
    for topic, msg, t in bag.read_messages():

        t_start = bag.get_start_time()
        # print(t_start)
        t_end = bag.get_end_time()
        t_i = t.to_sec() - t_start
        if topic == '/tf':
            if msg.transforms[0].child_frame_id == "Mallet":
                count_mallet += 1
                pose_i = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   msg.transforms[0].transform.translation.z,
                                   msg.transforms[0].transform.rotation.x,
                                   msg.transforms[0].transform.rotation.y,
                                   msg.transforms[0].transform.rotation.z,
                                   msg.transforms[0].transform.rotation.w,
                                   t_i])
                if len(mallet_poses) == 0 or not np.equal(np.linalg.norm(mallet_poses[-1][:2] - pose_i[:2]), 0):
                    mallet_poses.append(pose_i)

            elif msg.transforms[0].child_frame_id == "Puck":
                count_puck += 1
                pose_i = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   msg.transforms[0].transform.translation.z,
                                   msg.transforms[0].transform.rotation.x,
                                   msg.transforms[0].transform.rotation.y,
                                   msg.transforms[0].transform.rotation.z,
                                   msg.transforms[0].transform.rotation.w,
                                   t_i])
                if len(puck_poses) == 0 or not np.equal(np.linalg.norm(puck_poses[-1][:2] - pose_i[:2]), 0):
                    puck_poses.append(pose_i)

            elif msg.transforms[0].child_frame_id == "Table":
                count_table += 1
                pose_i = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   msg.transforms[0].transform.translation.z,
                                   msg.transforms[0].transform.rotation.x,
                                   msg.transforms[0].transform.rotation.y,
                                   msg.transforms[0].transform.rotation.z,
                                   msg.transforms[0].transform.rotation.w,
                                   t_i])
                if len(table_poses) == 0 or not np.equal(np.linalg.norm(table_poses[-1][2] - pose_i[2]), 0):
                    table_poses.append(pose_i)


    mallet_poses = np.array(mallet_poses)
    puck_poses = np.array(puck_poses)
    table_poses = np.array(table_poses)

    return puck_poses, mallet_poses, table_poses, t_end - t_start


def collision_filter():
    p.setCollisionFilterPair(puck, table, 0, 0, 1)
    p.setCollisionFilterPair(puck, table, 0, 1, 1)
    p.setCollisionFilterPair(puck, table, 0, 2, 1)
    p.setCollisionFilterPair(puck, table, 0, 4, 1)
    p.setCollisionFilterPair(puck, table, 0, 5, 1)
    p.setCollisionFilterPair(puck, table, 0, 6, 1)
    p.setCollisionFilterPair(puck, table, 0, 7, 1)

# relations
# slide: t_base (0)
# restitution and friction
#       t_down_rim_l (1),
#       t_down_rim_r (2),
#       t_left_rim (4),
#       t_right_rim (5),
#       t_up_rim_l (6),
#       t_up_rim_r (7)

bag = rosbag.Bag('./rosbag_data/2020-12-04-12-41-02.bag')
puck_poses, _, _, t = read_bag(bag)

# def filtposes(poses):
#     for i in range(poses.shape[0]):
#         poses[i, 3:6] = p.getEulerFromQuaternion(poses[i, 3:7])
#
#     b, a = signal.ellip(4, 0.0001, 100, 0.5)
#
#     sos_butter = signal.butter(4, 0.125, output='sos')
#
#     filtered_ellip = np.zeros((poses.shape))
#
#     filtered_butter = np.zeros((poses.shape))
#
#     for i in range(6):
#
#         filtered_ellip[:, i] = signal.filtfilt(b,a, poses[:,i])
#
#         filtered_butter[:, i] = signal.sosfiltfilt(sos_butter, poses[:,i])
#
#
#
#     return poses, filtered_ellip, filtered_butter
#
# def get_vel(poses):
#
#     t_series = np.linspace(0, t, poses.shape[0])
#     t_int_series = np.linspace(0, t, 10000)
#     puck_itpl= np.zeros((len(t_int_series), poses.shape[1]))
#     for i in range(8):
#         fpuck = interpolate.interp1d(t_series, poses[:, i], kind='linear')
#         puck_itpl[:, i] = fpuck(t_int_series)
#     slope = np.zeros((10000, 8))
#     h= 100
#     t_stueck = t / 10000.
#     for i in range(puck_itpl.shape[0]):  # Differenzenquotienten
#         if i > 9000:
#             slope[i,:] = np.zeros((1,8))
#             break
#         slope[i, :] = (puck_itpl[i + h, :] - puck_itpl[i, :]) / (h * t_stueck )
#
#
#     return puck_itpl, t_int_series, slope
#
# def slope2init(slope, t_series):
#     b, a = signal.ellip(4, 0.0001, 100, 0.9)
#     filterslope = np.zeros((slope.shape))
#     for i in range(6):
#         filterslope[:, i] = signal.filtfilt(b, a, slope[:, i])
#     initvel = np.zeros((1, 6))
#     idx = np.argmax(np.absolute(filterslope), axis=0)
#     for i, j in zip(idx, range(2)):
#         initvel[:,j] = filterslope[i, j]
#     initvel[0,2] = 0
#     initvel[0, 3:6] = filterslope[idx[0], 3:6]
#     plt.plot(t_series, filterslope[:, 0], label='filter')
#
#     return initvel, filterslope

def Diff(data, h):
    slope = np.zeros(data.shape)
    t_stueck = t / data.shape[0]
    for i in range(data.shape[0]):  # Differenzenquotienten
        if i > data.shape[0] - h - 1:
            slope[i, :] = np.zeros((1, data.shape[1]))
            break
        slope[i, :] = np.linalg.norm((data[i + h, :] - data[i, :]).reshape(-1,1), axis=1) / (h * t_stueck)


    return slope, np.linspace(0, t, data.shape[0])

def initdata(posedata):
    # posestart = posedata[354:, :]
    posestart = posedata
    posestart[:, 2] = 0.1172 * np.ones(posestart.shape[0])
    for i in range(posestart.shape[0]):
        posestart[i,3:6] = p.getEulerFromQuaternion(posestart[i,3:7])
    puck_posori = posestart[:,:6]
    puck_posori[:,3:5] = np.zeros(puck_posori[:,3:5].shape)

    return puck_posori


p.connect(p.GUI, 1234) # use build-in graphical user interface, p.DIRECT: pass the final results
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=150)
p.setTimeStep(1 / 240)
p.setGravity(0., 0., -9.81)
p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89, cameraTargetPosition=[1.55, 0.85, 1.])

file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "air_hockey_table", "model.urdf")
table = p.loadURDF(file, [1.7, 0.85, 0.117], [0, 0, 0.0, 1.0])
file = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "models", "puck", "model.urdf")
puck = p.loadURDF(file, puck_poses[0, 0:3], [0, 0, 0.0, 1.0])



puck_posori = initdata(puck_poses)
initvel, t_series =Diff(puck_posori, 40)

b, a = signal.butter(4, 0.09)
filtervel = np.zeros((initvel.shape))
for i in range(6):
    filtervel[:, i] = signal.filtfilt(b, a, initvel[:, i])
plt.plot(t_series, filtervel[:,0], label='x')
plt.plot(t_series, filtervel[:,1], label='y')
plt.plot(t_series, filtervel[:,2], label='z')
plt.plot(t_series, filtervel[:,3], label='wx')
plt.plot(t_series, filtervel[:,4], label='wy')
# plt.plot(t_series, filtervel[:,5], label='wz')
plt.ylim(0,4)
plt.legend()
plt.show()
readidx = 0

for linkidx in range(8):
    p.changeDynamics(table, linkidx, spinningFriction=0.01)
    p.changeDynamics(table, linkidx, restitution=0.845)

while True:

    while readidx != posestart.shape[0]:
        p.stepSimulation()
        if readidx == 0:
            lastpuck = posestart[readidx, 0:3]

        p.resetBasePositionAndOrientation(puck, posestart[readidx, 0:3], posestart[readidx, 3:7])
        p.addUserDebugLine(lastpuck, posestart[readidx, 0:3], lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3)
        lastpuck = posestart[readidx, 0:3]
        readidx += 1

    poses_pos = []
    poses_ang = []
    realvel = []
    readidx = 0
    p.setRealTimeSimulation(1)

    while readidx < 2000:
        p.resetBasePositionAndOrientation(puck, posestart[readidx, 0:3], posestart[readidx, 3:7])
        p.resetBaseVelocity(puck, linearVelocity=vel[0, :3], angularVelocity=vel[0, 3:6])
        p.stepSimulation()

        while readidx < 2000:

            collision_filter()
            if readidx == 0:
                lastpuck = posestart[readidx, 0:3]
                poses_pos.append(lastpuck)
                poses_ang.append(posestart[readidx, 3:7])
            realvel.append(p.getBaseVelocity(puck)[0])
            recordpos, recordang = p.getBasePositionAndOrientation(puck)
            poses_pos.append(recordpos)
            poses_ang.append(recordang)

            p.addUserDebugLine(lastpuck, recordpos, lineColorRGB= [0.1,0.1,0.5], lineWidth=5)

            lastpuck = recordpos
            readidx += 1
            # print(len(poses_pos), len(poses_ang))
            # time.sleep(1. / 240)




    # vel, _ = slope2init(np.asarry(poses_pos[:,0]))
    realvel = np.array(realvel)
    print('checkbagshape', np.array(vel).shape, 'checkrealshape', np.array(realvel).shape )
    print(np.linalg.norm(slope[:2000, :3] - realvel))
    # print(vel)
    plt.legend()
    plt.show()
