
from iiwa_envs.sim_With_bullet import *


def GUI_pre(bagdata, simdata, table_):
    p.connect(p.GUI, 1234)  # use build-in graphical user interface, p.DIRECT: pass the final results
    p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=-90.0, cameraPitch=-89,
                                 cameraTargetPosition=[1.55, 0.85, 1.])

    p.resetSimulation()
    p.setTimeStep(1 / 120.)

    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setGravity(0., 0., -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    path_robots = "/home/hszlyw/PycharmProjects/ahky/robots"
    file = os.path.join(path_robots, "models", "air_hockey_table", "model.urdf")
    table = p.loadURDF(file, list(table_[:3]), list(table_[3:]))
    file = os.path.join(path_robots, "models", "puck", "model2.urdf")
    file2 = os.path.join(path_robots, "models", "puck", "model3.urdf")

    readidx = 0
    lastpuck = np.hstack((bagdata[readidx, 1:3], 0.11945))
    lastpuck2 = np.hstack((simdata[readidx, 1:3], 0.11945))

    puck = p.loadURDF(file, lastpuck, [0, 0, 0.0, 1.0])
    puck2 = p.loadURDF(file2, lastpuck2, [0, 0, 0.0, 1.0])
    time.sleep(3)

    while readidx < bagdata.shape[0] - 1:
        p.resetBasePositionAndOrientation(puck, np.hstack((bagdata[readidx + 1, 1:3], 0.11945)), [0, 0, 0, 1.])
        p.resetBasePositionAndOrientation(puck2, np.hstack((simdata[readidx + 1, 1:3], 0.11945)), [0, 0, 0, 1.])
        p.stepSimulation()
        time.sleep(1/120.)
        # p.addUserDebugLine(lastpuck, np.hstack((bagdata[readidx + 1, 1:3], 0.11945)), lineColorRGB=[0.5, 0.5, 0.5],
        #                    lineWidth=3)
        # lastpuck = np.hstack((bagdata[readidx, 1:3], 0.11945))
        readidx += 1
    p.disconnect()


if __name__ == "__main__":

    bag_dir1 = "/home/hszlyw/Documents/airhockey/20210224/validate 2020/2020-11-20-12-35-23.bag"
    bag_dir2 = "/home/hszlyw/Documents/airhockey/20210224/all_long/5.txt" # 0, 5, 7, 12, 26, 31
    bag_dir3 = "/home/hszlyw/Documents/airhockey/20210224/all_short/8.txt" #2, 8, 11,
    bag_dir4 = "/home/hszlyw/Documents/airhockey/20210224/ori_all/2021-01-29-12-42-30.bag"




    bag = rosbag.Bag(os.path.join(bag_dir1))

    data = []
    bagdata1, table1 = read_bag(bag)
    # table = np.array([1.7, 0.85, 0.117, 0., 0., 0., 1.])
    table1 = np.array(table1)[0, :]
    # table = bagdata[0, :]
    table1[2] = 0.11945
    # bagdata = bagdata[1:, :]
    bagdata1[:, 3] = 0.11945 * np.ones(bagdata1.shape[0])


    # data = []
    # with open(bag_dir3, 'r') as f:
    #     # with open(bag_dir+'8.txt', 'r') as f:
    #
    #     for line in f:
    #         data.append(np.array(np.float64(
    #             line.replace("[", " ").replace("]", " ").replace(",", " ").replace("\n", "").split())))
    #     data = np.array(data)
    #     table1 = data[0, :]
    #     bagdata1 = np.array(data[1:, :])
    #     table1[2] = 0.11945
    #     bagdata1[:, 3] = 0.11945 * np.ones(bagdata1.shape[0])




    lin_ang_vel1 = get_vel(bagdata1.copy())
    for i, vel in enumerate(lin_ang_vel1):
        if (np.abs(vel[0:2]) > 0.1).any() and (np.abs(lin_ang_vel1[i + 2, 0:2]) > 0.1).any():
            begin_idx1 = i + 9
            break
        else:
            continue
    init_pos1 = np.hstack((bagdata1.copy()[0, 1:3], 0.11945))  # [3,]
    init_vel1 = vel2initvel(lin_ang_vel1, bagdata1.copy(), begin_idx1)  # In [n,7]





    p_all_dis_30 = [0.8811311846664506, 0.4105691544176001, 0.9124743731218027, 0.35900088632217547, 0.000000472,
                    0.000000609]  # best obj 0.1410335615239317
    p_all_dis_60 = [0.8505313619029673, 0.4771092354335756, 0.8940491783128354, 0.3356883524725823, 0.000000472,
                    0.000000609]  # best obj 0.1424837147294241
    p_all_dis_90 = [0.8834821676450403, 0.7644752860719204, 0.9499845801393345, 0.002574821936491776, 0.000000472,
                    0.000000609]  # best obj 0.14295354676680136
    p_seg_dis_33 = [0.8935860395431519, 0.11888450837805967, 0.8891523055307882, 0.053511844080132945,
                    5.236729810590324e-07, 0.007131593278764413]  # best obj 0.0522040600476066 0.03431201474734212
    p_seg_dis_66 = [0.9315619709374023, 0.14591225630032362, 0.7812911370537645, 0.12230765074491501,
                    9.06695006598454e-05, 0.006933054232588352]  # best obj 0.048745806677416195 0.03379604156229075

    p_seg_dis_99 = [0.9397830035465228, 0.11278709553712492, 0.7820095362422288, 0.1222818575716191,
                    9.111838291033335e-05, 0.007549535277524568]  # best obj 0.04878046485241181
    p_seg_ang_33 = [0.8191250007384172, 0.15159492324515833, 0.9499999064754436, 0.09399662911891937,
                    0.00029251963132992387, 0.0028112339024133717]  # best obj 3.7638519312853633 10.38573457698801
    p_seg_ang_66 = [0.9493355324671046, 0.106116896247992, 0.9499999799984453, 0.077954052871899,
                    8.175343042630247e-06, 3.9434919189378935e-06]  # best obj 3.630643092349588 7.849126801038876
    p_seg_ang_99 = [0.910131849261057, 0.106116896247992, 0.9499999799984453, 0.077954052871899,
                    8.175343042630247e-06, 3.9434919189378935e-06]  # best obj 2.939860393350947 8.239427807516991


    model = Model(p_seg_ang_99, init_pos1, init_vel1)
    t_sim1, sim_pos1, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata1 = np.hstack((t_sim1, sim_pos1))

    GUI_pre(bagdata1, simdata1, table1)













    model = Model(p_all_dis_90, init_pos1, init_vel1)
    t_sim1, sim_pos1, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata1 = np.hstack((t_sim1, sim_pos1))

    model = Model(p_all_dis_60, init_pos1, init_vel1)
    t_sim2, sim_pos2, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata2 = np.hstack((t_sim2, sim_pos2))

    model = Model(p_all_dis_90, init_pos1, init_vel1)
    t_sim3, sim_pos3, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata3 = np.hstack((t_sim3, sim_pos3))





    model = Model(p_seg_dis_33, init_pos1, init_vel1)
    t_sim4, sim_pos4, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata4 = np.hstack((t_sim4, sim_pos4))

    model = Model(p_seg_dis_66, init_pos1, init_vel1)
    t_sim5, sim_pos5, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata5 = np.hstack((t_sim5, sim_pos5))

    model = Model(p_seg_dis_99, init_pos1, init_vel1)
    t_sim6, sim_pos6, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata6 = np.hstack((t_sim6, sim_pos6))





    model = Model(p_seg_ang_33, init_pos1, init_vel1)
    t_sim7, sim_pos7, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata7 = np.hstack((t_sim7, sim_pos7))

    model = Model(p_seg_ang_66, init_pos1, init_vel1)
    t_sim8, sim_pos8, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata8 = np.hstack((t_sim8, sim_pos8))

    model = Model(p_seg_ang_99, init_pos1, init_vel1)
    t_sim9, sim_pos9, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata9 = np.hstack((t_sim9, sim_pos9))

    # simdatas_x = [simdata1[:200,1],simdata2[:200,1],simdata3[:200,1],simdata4[:200,1],simdata5[:200,1],simdata6[:200,1],simdata7[:200,1],simdata8[:200,1],simdata9[:200,1]]
    # simdatas_y = [simdata1[:200,2],simdata2[:200,2],simdata3[:200,2],simdata4[:200,2],simdata5[:200,2],simdata6[:200,2],simdata7[:200,2],simdata8[:200,2],simdata9[:200,2]]
    #
    #
    #
    #
    # N = 3
    #
    # data = (np.geomspace(1, 10, 100) + np.random.randn(N, 100)).T
    # cmap = plt.cm.coolwarm
    # rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
    # custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
    #                 Line2D([0], [0], color=cmap(.5), lw=4),
    #                 Line2D([0], [0], color=cmap(1.), lw=4)]
    #
    # fig, ax = plt.subplots()
    # # lines = ax.plot(data)
    # line1=ax.plot(simdata1[:200,1], simdata1[:200,2])
    # line2=ax.plot(simdata2[:200,1], simdata2[:200,2])
    # line3=ax.plot(simdata3[:200,1], simdata3[:200,2])
    # line4=ax.plot(simdata4[:200,1], simdata4[:200,2])
    # line5=ax.plot(simdata5[:200,1], simdata5[:200,2])
    # line6=ax.plot(simdata6[:200,1], simdata6[:200,2])
    # line7=ax.plot(simdata7[:200,1], simdata7[:200,2])
    # line8=ax.plot(simdata8[:200,1], simdata8[:200,2])
    # line9=ax.plot(simdata9[:200,1], simdata9[:200,2])
    # line10= ax.plot(bagdata1[:200, 1], bagdata1[:200, 2],label='orig', color='k' )
    # ax.legend(custom_lines, ['overall dataset, distance loss', 'partial dataset, distance loss', 'partial dataset, angular loss', 'orig'])
    #
    #
    #
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0.7,3])
    ax.set_ylim([0.3,1.45])
    ax.plot(bagdata1[:-500, 1], bagdata1[:-500, 2], label='original data', marker='.', alpha=1,markersize = 2.2, color='k')
    ax.scatter(bagdata1[0, 1], bagdata1[0, 2], label='start point', marker='o', s=15,color='r')
    ax.plot(simdata1[:200, 1], simdata1[:200, 2], label='overall distance loss', marker='o',markersize = 1,alpha=0.7, color='b')
    ax.plot(simdata2[:200, 1], simdata2[:200, 2], marker='o',markersize = 1,alpha=0.4, color='b')
    ax.plot(simdata3[:200, 1], simdata3[:200, 2],  marker='o',markersize = 1,alpha=0.4, color='b')
    ax.plot(simdata4[:200, 1], simdata4[:200, 2], label='partial distance loss', marker='o',markersize = 1,alpha=0.7, color='r')
    ax.plot(simdata5[:200, 1], simdata5[:200, 2], marker='o',markersize = 1,alpha=0.4, color='r')
    ax.plot(simdata6[:200, 1], simdata6[:200, 2], marker='o',markersize = 1,alpha=0.4, color='r')
    ax.plot(simdata7[:200, 1], simdata7[:200, 2], label='partial angular loss', marker='o',markersize = 1,alpha=0.7, color='y')
    ax.plot(simdata8[:200, 1], simdata8[:200, 2],  marker='o',markersize = 1,alpha=0.4, color='y')
    ax.plot(simdata9[:200, 1], simdata9[:200, 2],  marker='o',markersize = 1,alpha=0.4, color='y')







    fig.tight_layout()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(1, 0.9))



    # plt.savefig('l_one.png', format='png')
    plt.show()

    None

