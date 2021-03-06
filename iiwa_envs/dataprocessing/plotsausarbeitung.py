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
    return stamp[1]

if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/20210224/all_long/"
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
    bagdataset1=[]
    bagdataset2=[]

    for nn, filename in enumerate(dir_list):
        # filename = dir_list[1]
        # filename = "2021-02-24-15-09-02.bag"

        # print(filename)
        data = []
        with open(bag_dir + filename, 'r') as f:
            # with open(bag_dir+'8.txt', 'r') as f:

            for line in f:
                data.append(np.array(np.float64(
                    line.replace("[", " ").replace("]", " ").replace(",", " ").replace("\n", "").split())))
            data = np.array(data)
            table = data[0, :]
            bagdata = np.array(data[1:, :])
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

        parameters = [0.89999997588924,0.0862029979358403, 0.8999999761581421, 0.0117223169654607771]  # loss is angle
        # parameters = [0.8999985117910535, 0.08618541772442043,0.699999988079071, 0.4000000059604645] # loss is dis
        shortparams= [0.699999988079071, 0.4000000059604645, ]
        # parameters = [ 0.8254059088662781,0.10319242853023827, 0.9299991355795136,0.014490739442408085, 0.010485903276094338]
        parameters_comb = [0.5692870164606006, 0.5728446366428586,0.7815188983391684, 0.7779911832945189, 9.459688616606632e-06, 0.06995505922640383]
        parameters_puredis = [0.5000000333785332, 0.47638183117953, 0.015380650174596779, 0.751316530185423, .001, 0.07833489775657654]
        parameters_ang2dis1 = [0.6588862055555563, 0.5588022042031872,0.7120770515310699,0.48903109637226916, 0.005036661922042332, 0.0055209567542747985]
        parameters_pureang= [0.5004560623165386, 0.5697433172992706,0.9146034553520569,0.7792704235951046,0.006211612598792474, 0.04577464769528256]
        parameters_angtimesdis= [0.5016025459847976, 0.4586099838084054, 0.681699731320961, 0.13918265930921014,0.000000472,  0.000000609]
        params_27bagsmult = [0.5350643232102498, 0.019638493115294142, 0.6041054989098973, 0.07778764783505487, 0.001162435774030039,0.0008092878072303286]
        params_27bagsmult0003ex=[0.5948782061362958, 0.6767296591391992, 0.74488450111527, 0.7573606599450943, 0.014027191373899347, 0.0004118297628308313]
        params_aus1 = [0.7322243090496859, 0.40040546232180796, 0.7182018939691728, 0.3882585465536242, 0.008124363608658314, 0.0009918306660691937]
        params_aus2 = [0.7154416441917419, 0.773118393744553, 0.7803165912628174, 2.195339187958629e-13, 0.001052391016855836, 0.0009999909671023488]
        param_aus3_dis_long = [ 0.929532964993739, 0.13050455779452633, 0.8377456665039062, 0.12618421915495376, 0.0024844819473709067, 0.0004477310517563276]
        param_aus4_ang_long = [0.7781064712815549, 0.7999377967389442, 0.9300000000991063, 0.05591317113149914, 8.882570546120405e-05, 0.00018898608384461957]
        param_aus3_dis_short = [0.8377456665039062, 0.12618421915495376]
        param_aus4_dis_short = [0.9300000000991063, 0.05591317113149914]

        # param_aus5_short_dis =
        # param_aus6_short_ang =
        model = Model(param_aus3_dis_long, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))   # [n,4]
        endidx=get_collide_point(simdata.copy())
        simdataset1.append(simdata[:endidx, :])
        bagdataset1.append(bagdata)

        model = Model(param_aus4_ang_long, init_pos, init_vel)
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
        simdata = np.hstack((t_sim, sim_pos))   # [n,4]
        endidx=get_collide_point(simdata.copy())
        simdataset2.append(simdata[:endidx, :])


    fig = plt.figure()
    # st = fig.suptitle("One collision with short rims", fontsize="x-large")
    for i in range(nn):
        # ax1 = fig.add_subplot(6,4,i+1)
        # ax1.plot(bagdataset1[i][:,1], bagdataset1[i][:,2], label='rosbag trajectory segment' ,marker= '+',markersize=.02 )
        # ax1.scatter(bagdataset1[i][0,1], bagdataset1[i][0,2], marker= 'o',s=10,color='red' )
        # ax1.plot(simdataset1[i][:,1], simdataset1[i][:,2], label='simulation trajectory segment, obj1', marker= '.',markersize=.02)
        # ax1.plot(simdataset2[i][:,1], simdataset2[i][:,2], label='simulation trajectory segment, obj2', marker= '^',markersize=.02)
        ax1 = fig.add_subplot(6,4,i+1)
        start_time= bagdataset1[i][0,0]
        ax1.scatter(bagdataset1[i][:,0] - start_time, bagdataset1[i][:,1], label='rosbag x position with time' ,marker= '.', s=2, color='g')
        ax1.scatter(np.linspace(0, simdataset1[i].shape[0] / 120., len(simdataset1[i][:,1])), simdataset1[i][:,1], label='obj1 x position with time' ,marker= '.', s=1 )
        ax1.scatter(np.linspace(0, simdataset2[i].shape[0] / 120., len(simdataset2[i][:,1])), simdataset2[i][:,1], label='obj2 x position with time' ,marker= '.', s=1 )

    fig.tight_layout()
    lines, labels = fig.axes[-1].get_legend_handles_labels()

    fig.legend(lines, labels, loc='lower right')
    plt.savefig('l_one.eps', format='eps')
    plt.show()

    None

