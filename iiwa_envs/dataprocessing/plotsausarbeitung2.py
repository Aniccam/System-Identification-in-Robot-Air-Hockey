from iiwa_envs.sim_With_bullet import *


def get_collide_point(bagdata):

    bagdata = np.around(bagdata, 2)
    h = 3
    stamp = []
    delta = []

    for i in range(bagdata.shape[0]):
        if i < 0:
            pass
        elif i >= bagdata.shape[0] - h:
            break
        else:
            if  ((  (bagdata[i, 1:3] - bagdata[i-h, 1:3])  )*
                    (  (bagdata[i+h, 1:3] - bagdata[i, 1:3])    ) < 0).any():

                delta.append([i,   (bagdata[i, 1:3] - bagdata[i-h, 1:3])  *
                     (bagdata[i+h, 1:3] - bagdata[i, 1:3]) ])
            else:
                pass

    delta = np.array(delta)

    for i in range(1, delta.shape[0]):
        if delta[i, 0] - delta[i-1, 0] < 10:
            pass
        else:
            stamp.append(delta[i, 0])
    stamp = np.array(stamp)
    if len(stamp) >= 2:
        return stamp[1]
    else:
        return stamp[0] + 20
if __name__ == "__main__":

    bag_dir1 = "/home/hszlyw/Documents/airhockey/20210224/validate 2020/"
    bag_dir2 = "/home/hszlyw/Documents/airhockey/all_long/"
    bag_dir3 = "/home/hszlyw/Documents/airhockey/20210224/validate 2020/"

    # bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited/bagfiles/"
    dir_list = os.listdir(bag_dir)
    dir_list.sort()
    nn = len(dir_list)


    # choose bag file
    # filename = dir_list[6]
    count = 0
    Loss = 0
    # fig, ax = plt.subplot(6,4, figsize=(10,8))
    simdataset1=[]
    simdataset2=[]
    simdataset3=[]
    simdataset4=[]
    simdataset5=[]

    bagdataset1=[]
    bagdataset2=[]
    loss1= []
    loss2= []
    loss3=[]
    loss4=[]
    loss5=[]


    for id, filename in enumerate(dir_list):
        # filename = dir_list[1]
        # filename = "2021-02-24-15-09-02.bag"

        # print(filename)
        bag = rosbag.Bag(os.path.join(bag_dir, filename))

        data = []
        bagdata, table = read_bag(bag)
        # table = np.array([1.7, 0.85, 0.117, 0., 0., 0., 1.])
        table = np.array(table)[0, :]
        # table = bagdata[0, :]
        table[2] = 0.11945
        # bagdata = bagdata[1:, :]
        bagdata[:, 3] = 0.11945 * np.ones(bagdata.shape[0])
        lin_ang_vel = get_vel(bagdata.copy())
        # init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]
        for i, vel in enumerate(lin_ang_vel):
            if (np.abs(vel[0:2]) > 0.1).any() and (np.abs(lin_ang_vel[i + 2, 0:2]) > 0.1).any():
                begin_idx = i + 10
                break
            else:
                continue
        init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]
        init_vel = vel2initvel(lin_ang_vel, bagdata.copy(), begin_idx)  # In [n,7]

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

        # param_aus5_short_dis =
        # param_aus6_short_ang =
        model = Model(params_aus1, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))   # [n,4]
        # endidx=get_collide_point(simdata.copy())
        simdataset1.append(simdata[:200,:])
        bagdataset1.append(bagdata)
        loss1.append(get_Err(bagdata, simdata)/5.)


        model = Model(params_aus2, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))   # [n,4]
        # endidx=get_collide_point(simdata.copy())
        simdataset2.append(simdata[:200,:])
        loss2.append(get_Err(bagdata, simdata)/5.)

        model = Model(params_aus_w1, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))  # [n,4]
        # endidx = get_collide_point(simdata.copy())
        simdataset3.append(simdata[:200,:])
        loss3.append(get_Err(bagdata, simdata)/5.)

        model = Model(param_aus3_dis_long, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))  # [n,4]
        # endidx = get_collide_point(simdata.copy())
        simdataset4.append(simdata[:200, :])
        loss4.append(get_Err(bagdata, simdata) / 5.)

        model = Model(param_aus4_ang_long, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))  # [n,4]
        # endidx = get_collide_point(simdata.copy())
        simdataset5.append(simdata[:200, :])
        loss5.append(get_Err(bagdata, simdata) / 5.)

    print("1:", np.sum(loss1))
    print("2:", np.sum(loss2))
    print("3:", np.sum(loss3))
    print("4:", np.sum(loss4))
    print("5:", np.sum(loss5))

    # x = ['cmp_dis', 'cmp_ang', 'comp_dis_ang','sgm_dis','sgm_ang']
    #
    # y = [np.sum(loss1), np.sum(loss2), np.sum(loss3),np.sum(loss4),np.sum(loss5)]
    #
    # plt.barh(x, y)
    #
    # for index, value in enumerate(y):
    #     plt.text(value, index, str(value))
    # # plt.show()








    fig = plt.figure()
    # st = fig.suptitle("One collision with short rims", fontsize="x-large")
    for i in range(nn):
        ax1 = fig.add_subplot(3,2,i+1)
        ax1.scatter(bagdataset1[i][:,1], bagdataset1[i][:,2], label='rosbag trajectory segment' ,marker= '.',s=.4, color='g', alpha=.3 )
        ax1.scatter(bagdataset1[i][0,1], bagdataset1[i][0,2], marker= 'o',s=5,color='r' )
        for jj in range(simdataset1[i].shape[0]):
            if jj <= 0:
                ax1.scatter(simdataset1[i][:,1], simdataset1[i][:,2], label='params= comp_dis', marker= '.',s=.1, alpha=1-jj*.001,color='b')
            else:
                ax1.scatter(simdataset1[i][:,1], simdataset1[i][:,2], marker= '.',s=.1, alpha=0,color='b')

        for jj in range(simdataset2[i].shape[0]):
            if jj <= 0:
                ax1.scatter(simdataset2[i][:,1], simdataset2[i][:,2], label='params= comp_ang', marker= '^',s=.1, alpha=1-jj*.001,color='y')
            else:
                ax1.scatter(simdataset2[i][:,1], simdataset2[i][:,2], marker= '^',s=.1, alpha=0,color='y')

        for jj in range(simdataset3[i].shape[0]):
            if jj <= 0:
                ax1.scatter(simdataset3[i][:,1], simdataset3[i][:,2], label='params= comp_dis_ang', marker= '^',s=.1, alpha=1-jj*.001,color='m')
            else:
                ax1.scatter(simdataset3[i][:,1], simdataset3[i][:,2], marker= '^',s=.1, alpha=0,color='m')
        for jj in range(simdataset4[i].shape[0]):
            if jj <= 0:
                ax1.scatter(simdataset4[i][:,1], simdataset4[i][:,2], label='params= sgm_dis', marker= '^',s=.1, alpha=1-jj*.001,color='k')
            else:
                ax1.scatter(simdataset4[i][:,1], simdataset4[i][:,2], marker= '^',s=.1, alpha=0,color='k')

        for jj in range(simdataset5[i].shape[0]):
            if jj <= 0:
                ax1.scatter(simdataset5[i][:,1], simdataset5[i][:,2], label='params= sgm_ang', marker= '^',s=.1, alpha=1-jj*.001,color='c')
            else:
                ax1.scatter(simdataset5[i][:,1], simdataset5[i][:,2], marker= '^',s=.1, alpha=0,color='c')


        # ax1 = fig.add_subplot(6,4,i+1)
        # start_time= bagdataset1[i][0,0]
        # ax1.scatter(bagdataset1[i][:,0] - start_time, bagdataset1[i][:,1], label='rosbag x position with time' ,marker= '.', s=2, color='g')
        # ax1.scatter(np.linspace(0, simdataset1[i].shape[0] / 120., len(simdataset1[i][:,1])), simdataset1[i][:,1], label='obj1 x position with time' ,marker= '.', s=1 )
        # ax1.scatter(np.linspace(0, simdataset2[i].shape[0] / 120., len(simdataset2[i][:,1])), simdataset2[i][:,1], label='obj2 x position with time' ,marker= '.', s=1 )
        #
    fig.tight_layout()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower right')



    plt.savefig('l_one.png', format='png')
    plt.show()

    None

