
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
from matplotlib.lines import Line2D


from iiwa_envs.sim_With_bullet import *


if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/20210224/validate 2020/"
    dir_list = os.listdir(bag_dir)
    dir_list.sort()

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

    params= [p_all_dis_30,p_all_dis_60,p_all_dis_90,p_seg_dis_33,p_seg_dis_66,p_seg_dis_99,p_seg_ang_33,p_seg_ang_66,p_seg_ang_99]
    Loss = []
    for param in params:
        loss=[]
        for filename in dir_list:

            bag = rosbag.Bag(os.path.join(bag_dir+filename))

            data = []
            bagdata, table = read_bag(bag)
            # table = np.array([1.7, 0.85, 0.117, 0., 0., 0., 1.])
            table = np.array(table)[0, :]
            # table = bagdata[0, :]
            table[2] = 0.11945
            # bagdata = bagdata[1:, :]
            bagdata[:, 3] = 0.11945 * np.ones(bagdata.shape[0])
            lin_ang_vel = get_vel(bagdata.copy())
            for i, vel in enumerate(lin_ang_vel):
                if (np.abs(vel[0:2]) > 0.1).any() and (np.abs(lin_ang_vel[i + 2, 0:2]) > 0.1).any():
                    begin_idx1 = i + 9
                    break
                else:
                    continue
            init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]
            init_vel = vel2initvel(lin_ang_vel, bagdata.copy(), begin_idx1)  # In [n,7]



            model = Model(param, init_pos, init_vel)
            t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')  # [n,] [n,3] [n,3]
            simdata = np.hstack((t_sim, sim_pos))
            loss.append( get_Err(bagdata, simdata) )
        Loss.append(np.mean(loss))

    print(np.mean(Loss[:3]), np.mean(Loss[3:6]),np.mean(Loss[6:9])  )






