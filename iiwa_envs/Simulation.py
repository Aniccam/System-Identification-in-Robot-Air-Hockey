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

def filter(data, Wn=0.1):
    b, a = signal.butter(6, Wn)  # Wn is the frequency when amplitude drop of 3 dB, smaller Wn, smaller frequency pass

    for i in range(data.shape[1]):
        data[:, i] = signal.filtfilt(b, a, data[:, i], method='gust')
    return data

def Diff(data, h):
    slope = np.zeros((data.shape))
    # data_ = np.zeros((data.shape[0], 2))

#   data changed after using, so .copy is necessary
#   tradeoff: large Wn, good coincidence with original traj, position noised. small Wn, bad coincidence with ori. traj at corner, position less noised
#   in order to get initial velocity, here ignore the corer precision with small Wn
    data2 = filter(data.copy(), 0.5)

    # data_[:, 0] = np.linalg.norm(data[:, :2], axis=1)
    # data_[:, 1] = data[:, 2]   # position already exist noise





    t_stueck = t / data.shape[0]
    for i in range(data.shape[0]):  # Differenzenquotienten
        if i > data.shape[0] - h - 1:
            slope[i, :] = np.zeros((1, data.shape[1]))
            break
        slope[i, :] = (data[i + h, :] - data[i, :]) / (h * t_stueck)


    slope2 = filter(slope.copy(), 0.5)
    slope3 = filter(slope.copy(), 0.1)
    slope4 = filter(slope.copy(), 0.085)
    slope5 = filter(slope.copy(), 0.070)

# for stoppoint before return:
    # plt.plot(np.linspace(0, t, data.shape[0]), slope[:,0], ls='-', label='raw', color='r'),  plt.plot(np.linspace(0, t, data.shape[0]), slope2[:,0],ls='-', label='Wn=0.5', color='b'),  plt.plot(np.linspace(0, t, data.shape[0]), slope3[:,0],ls='-', label='Wn=0.1', color='y'), plt.plot(np.linspace(0, t, data.shape[0]), slope4[:,0],ls='-', label='Wn=0.05', color='k'), plt.plot(np.linspace(0, t, data.shape[0]), slope5[:,0],ls='-', label='Wn=0.01', color='purple')

    return slope, np.linspace(0, t, data.shape[0])


def initdata(posedata):
    posestart = posedata[354:, :]
    # posestart = posedata
    posestart[:, 2] = 0.1172 * np.ones(posestart.shape[0])
    for i in range(posestart.shape[0]):
        posestart[i,3:6] = p.getEulerFromQuaternion(posestart[i,3:7])
    puck_posori = posestart[:,:6]
    puck_posori[:,3:5] = np.zeros(puck_posori[:,3:5].shape)

    puck_posori2 = np.concatenate((puck_posori[:, :2], puck_posori[:, 5:]), axis=1)
    data_ = np.zeros((puck_posori2.shape[0], 2))

    data_[:, 0] = np.linalg.norm(puck_posori2[:, :2], axis=1)
    data_[:, 1] = puck_posori2[:, 2]   # position already exist noise

    return puck_posori, data_

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



puck_posori_6, puck_posori_2 = initdata(puck_poses)
speed, t_series =Diff(puck_posori_2, 10)

#init plot
pick = [0,1,5]
label = ('x', 'y', 'z', 'wx', 'wy', 'wz')
color = ('r','g','b','k','y','g')
# vel = np.zeros(len(pick),)
# fig, axes = plt.subplots(len(pick), 1)
# def bagplotvel():
#     # getori =
#     for i in pick:
#         idx = np.argmax(filtervel[:, p])
#         vel[i] = filtervel[idx, p]
#         axes[i].scatter(t_series[idx], vel[i], edgecolors='red', s=20)
#         axes[i].text(t_series[idx]+1.5, vel[i]-0.8, s='('+str(idx)+','+str(round(vel[i],2))+')', color='purple')
#
#
#         axes[i].plot(t_series, filtervel[:, p], label=label[p], color=color[i])
#     fig.legend()
#     return vel

# vel = bagplotvel()

posestart = puck_poses[354:, :]
readidx = 0
tan_theta = (posestart[25, :2] - posestart[0, :2])[1] / (posestart[25, :2] - posestart[0, :2])[0]
cos_theta = 1 / np.sqrt(np.square(tan_theta) + 1)
sin_theta = tan_theta / np.sqrt(np.square(tan_theta) + 1)
initori = [cos_theta, sin_theta]

linvel = np.zeros((2,))
for i, s in enumerate(speed[:,0]):
    if s> 2.9:
        linvel = speed[i,0]
        angvel = speed[i,1]
        break

init_linvel = np.hstack(( linvel * np.array(initori), 0 ))
init_angvel = np.hstack(([0,0], angvel))


for linkidx in range(8):
    p.changeDynamics(table, linkidx, spinningFriction=0.01)
    p.changeDynamics(table, linkidx, restitution=0.845)



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
# p.setPhysicsEngineParameter(fixedTimeStep=t_series[-1]/len(t_series))
p.resetBasePositionAndOrientation(puck, posestart[readidx, 0:3], posestart[readidx, 3:7])
p.resetBaseVelocity(puck, linearVelocity=init_linvel, angularVelocity=init_angvel)
p.stepSimulation()

while readidx < speed.shape[0]:

    collision_filter()
    if readidx == 0:
        lastpuck = posestart[readidx, 0:3]
        poses_pos.append(lastpuck)
        poses_ang.append(posestart[readidx, 3:7])
    realvel.append(p.getBaseVelocity(puck)[0] + p.getBaseVelocity(puck)[1])
    recordpos, recordang = p.getBasePositionAndOrientation(puck)
    poses_pos.append(recordpos)
    poses_ang.append(recordang)

    p.addUserDebugLine(lastpuck, recordpos, lineColorRGB= [0.1,0.1,0.5], lineWidth=5)

    lastpuck = recordpos
    readidx += 1


    # print(len(poses_pos), len(poses_ang))
    # time.sleep(1. / 240)


fig, axes = plt.subplots(len(pick), 1)

def pltvel(vel, classify):
    # getori =
    for i,p in enumerate(pick): # 0 1 5
        # vel[i] = vel[idx, i]
        # axes[i].scatter(t_series[idx], vel[i], edgecolors='red', s=20)
        # axes[i].text(t_series[idx]+1.5, vel[i]-0.8, s='('+str(idx)+','+str(round(vel[i],2))+')', color='purple')
        if classify == 'bag':
            axes[i].plot(t_series, vel[:, p], label=classify + label[i], color=color[i])
        elif classify == 'real':
            axes[i].plot(t_series, vel[:, p], label=classify + label[i], color=color[i], alpha=0.4)



bagvel,_ = Diff(puck_posori_6, 10)
pltvel(bagvel, 'bag')
pltvel(np.array(realvel),'real')

# realvel = np.array(realvel)
# print('checkbagshape', np.array(vel).shape, 'checkrealshape', np.array(realvel).shape )
# for i in pick:
#     axes[i].plot(t_series, realvel[:, i])
fig.legend()
plt.show()
