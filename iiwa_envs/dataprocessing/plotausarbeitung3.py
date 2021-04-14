
from iiwa_envs.sim_With_bullet import *


if __name__ == "__main__":

    bag_dir1 = "/home/hszlyw/Documents/airhockey/20210224/validate 2020/2020-11-20-12-38-13.bag"
    bag_dir2 = "/home/hszlyw/Documents/airhockey/20210224/all_long/24.txt"
    bag_dir3 = "/home/hszlyw/Documents/airhockey/20210224/all_short/2.txt"



    # filename = dir_list[1]
    # filename = "2021-02-24-15-09-02.bag"

    # print(filename)
    bag = rosbag.Bag(os.path.join(bag_dir1))

    data = []
    bagdata1, table1 = read_bag(bag)
    # table = np.array([1.7, 0.85, 0.117, 0., 0., 0., 1.])
    table1 = np.array(table1)[0, :]
    # table = bagdata[0, :]
    table1[2] = 0.11945
    # bagdata = bagdata[1:, :]
    bagdata1[:, 3] = 0.11945 * np.ones(bagdata1.shape[0])
    lin_ang_vel1 = get_vel(bagdata1.copy())
    for i, vel in enumerate(lin_ang_vel1):
        if (np.abs(vel[0:2]) > 0.1).any() and (np.abs(lin_ang_vel1[i + 2, 0:2]) > 0.1).any():
            begin_idx1 = i + 9
            break
        else:
            continue
    init_pos1 = np.hstack((bagdata1.copy()[0, 1:3], 0.11945))  # [3,]
    init_vel1 = vel2initvel(lin_ang_vel1, bagdata1.copy(), begin_idx1)  # In [n,7]




    data = []
    with open(bag_dir2, 'r') as f:
        # with open(bag_dir+'8.txt', 'r') as f:

        for line in f:
            data.append(np.array(np.float64(
                line.replace("[", " ").replace("]", " ").replace(",", " ").replace("\n", "").split())))
        data = np.array(data)
        table2 = data[0, :]
        bagdata2 = np.array(data[1:, :])
        table2[2] = 0.11945
        bagdata2[:, 3] = 0.11945 * np.ones(bagdata2.shape[0])
    lin_ang_vel2 = get_vel(bagdata2.copy())
    for i, vel in enumerate(lin_ang_vel2):
        if (np.abs(vel[0:2]) > 0.1).any() and (np.abs(lin_ang_vel2[i + 2, 0:2]) > 0.1).any():
            begin_idx2 = i + 9
            break
        else:
            continue
    init_pos2 = np.hstack((bagdata2.copy()[0, 1:3], 0.11945))  # [3,]
    init_vel2 = vel2initvel(lin_ang_vel2, bagdata2.copy(), begin_idx2)  # In [n,7]

    data = []
    with open(bag_dir3, 'r') as f:
        # with open(bag_dir+'8.txt', 'r') as f:

        for line in f:
            data.append(np.array(np.float64(
                line.replace("[", " ").replace("]", " ").replace(",", " ").replace("\n", "").split())))
        data = np.array(data)
        table3 = data[0, :]
        bagdata3 = np.array(data[1:, :])
        table3[2] = 0.11945
        bagdata3[:, 3] = 0.11945 * np.ones(bagdata3.shape[0])
    lin_ang_vel3 = get_vel(bagdata3.copy())
    for i, vel in enumerate(lin_ang_vel3):
        if (np.abs(vel[0:2]) > 0.1).any() and (np.abs(lin_ang_vel3[i + 2, 0:2]) > 0.1).any():
            begin_idx3 = i + 10
            break
        else:
            continue
    init_pos3 = np.hstack((bagdata3.copy()[0, 1:3], 0.11945))  # [3,]
    init_vel3 = vel2initvel(lin_ang_vel3, bagdata3.copy(), begin_idx3)  # In [n,7]



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
    p_seg_ang_99 = [0.7990131849261057, 0.1842469905670424, 0.9499999875915864, 0.13122795522212982,
                    0.0001241783920460382, 0.008661260161902341]  # best obj 2.939860393350947 8.239427807516991

    model = Model(p_all_dis_60, init_pos1, init_vel1)
    t_sim1, sim_pos1, _ = model.sim_bullet(table1, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata1 = np.hstack((t_sim1, sim_pos1))

    model = Model(p_seg_dis_66, init_pos2, init_vel2)
    t_sim2, sim_pos2, _ = model.sim_bullet(table2, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata2 = np.hstack((t_sim2, sim_pos2))

    model = Model(p_seg_dis_66, init_pos3, init_vel3)
    t_sim3, sim_pos3, _ = model.sim_bullet(table3, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata3 = np.hstack((t_sim3, sim_pos3))

    model = Model(p_seg_ang_66, init_pos2, init_vel2)
    t_sim4, sim_pos4, _ = model.sim_bullet(table2, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata4 = np.hstack((t_sim4, sim_pos4))

    model = Model(p_seg_ang_66, init_pos3, init_vel3)
    t_sim5, sim_pos5, _ = model.sim_bullet(table3, 'DIRECT')  # [n,] [n,3] [n,3]
    simdata5 = np.hstack((t_sim5, sim_pos5))


    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    # ax.set_xlim([0.7,3])
    # ax.set_ylim([0.3,1.45])
    # ax.plot(bagdata1[:, 1], bagdata1[:, 2], label='original data', markersize = 1, marker='.', alpha=1)
    # ax.scatter(bagdata1[0, 1], bagdata1[0, 2], label='start point', marker='o', s=15,color='r')
    # ax.plot(simdata1[:270, 1], simdata1[:270, 2], label='simulation', marker='o',markersize = 1)
    # ax.title.set_text('overall movement')



    ax = fig.add_subplot(3, 1, 1)
    ax.set_xlim([0.7,3])
    ax.set_ylim([0.3,1.45])
    ax.plot(bagdata2[:, 1], bagdata2[:, 2], label='original data', marker='.', alpha=1,markersize = 1, color='k')
    ax.scatter(bagdata2[0, 1], bagdata2[0, 2], label='start point', marker='o', s=15,color='r')
    ax.plot(simdata2[:400, 1], simdata2[:400, 2], label='distance error as loss', marker='o',markersize = 1)
    ax.plot(simdata4[:400, 1], simdata4[:400, 2], label='angular error as lossg', marker='o',markersize = 1)

    ax.title.set_text('long rim collision')




    ax = fig.add_subplot(3, 1, 2)
    ax.set_xlim([0.7,3])
    ax.set_ylim([0.3,1.45])
    ax.plot(bagdata3[:, 1], bagdata3[:, 2], label='original data', marker='.',alpha=1,markersize = 1, color='k')
    ax.scatter(bagdata3[0, 1], bagdata3[0, 2], label='start point', marker='o', s=15,color='r')
    ax.plot(simdata3[:150, 1], simdata3[:150, 2], label='simulation using distance error as loss', marker='o',markersize = 1)
    ax.plot(simdata5[:150, 1], simdata5[:150, 2], label='simulation using angular error as loss', marker='o',markersize = 1)

    ax.title.set_text('short rim collision')


    fig.tight_layout()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(.6, 0.3))



    # plt.savefig('l_one.png', format='png')
    plt.show()

    None

