import pandas as pd
import numpy as np
from bo.design_space.design_space import DesignSpace
from bo.optimizers.hebo import HEBO
from sim_With_bullet import *
import csv
def obj(params, table) -> np.ndarray:
    out = np.zeros((params.shape[0], 1))
    for i in range(params.shape[0]):
        X = [
             params.iloc[i].get('left_right_rim_res'),
             params.iloc[i].get('left_right_rim_f'),


             ]
        model = Model(X, init_pos.copy(), init_vel.copy())
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')
        simdata = np.hstack((t_sim, sim_pos))
        # loss = Lossfun(bagdata.copy(), simdata, 'DIRECT')
        loss = Lossfun2(bagdata.copy(), simdata.copy(), 'DIRECT')
        out[i] = loss
    return out



if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited/dataset_long/"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()


#  [t_lateral_f, left_right_rim_res, left_rim_f, four_side_rim_res, four_side_rim_latf, angvel]  [5,]
    param_list = [
        {'name': 'left_right_rim_res', 'type': 'num', 'lb': .4, 'ub': .9},
        {'name': 'left_right_rim_f', 'type': 'num', 'lb': 0., 'ub': .7}
    ]

    space = DesignSpace().parse(param_list)
    opt = HEBO(space)

    # init value
    rec = pd.DataFrame({

                        'left_right_rim_res': [.9, .7, .5, .4, .67, .75],
                        'left_right_rim_f': [.1, .3, .5, .2, .01, .4]


                        }
                        )


    #
    # filename = "24.txt"
    # print(filename)
    data = []

    objs = []
    params = []

    for i in range(2):
        rec = opt.suggest(n_suggestions=4)
        obj_value = np.zeros((4, 1)) # nrows = n_suggestions

        for filename in dir_list:
            # filename = "0_.txt"
            # print(filename)
            with open(bag_dir + filename, 'r') as f:
                for line in f:
                    data.append(np.array(np.float64(line.replace("[", " ").replace("]", " ").replace(",", " ").split())))
                bagdata = np.array(data)
                table = bagdata[0, :]
                table[2] = 0.11945
                bagdata = bagdata[1:, :]
                bagdata[:, 3] = 0.11945 * np.ones(bagdata.shape[0])
                lin_ang_vel = get_vel(bagdata.copy())  # In [n,7], Out[n,6]
                init_vel = lin_ang_vel[0, :]

                # bagdata = bagdata[:back_process(lin_ang_vel), :]
                init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]

                tempobj = obj(rec, table)
                # while (tempobj == 180.3334).any():  # 180.3334 wrong loss, should remove
                #     tempobj = np.delete(tempobj, np.where(tempobj == 180.3334))

                # tempobj = np.mean(tempobj)

                obj_value = tempobj + obj_value       # float64


        print(obj_value)
        opt.observe(rec, obj_value)
        objs.append(obj_value)


        # print('iter', i)
        #
        # print("Current: ", "objective: {} ".format(obj_value[0, 0]),
        #       "parameter: [left_right_rim_res: {}, "
        #       "left_right_rim_latf: {},]".format(rec.left_right_rim_res.values[0],
        #                rec.left_right_rim_f.values[0]))
        # # rec.t_lateral_f.values[0], rec.left_right_rim_res.values[0], rec.left_right_rim_f.values[0],
        # # t_lateral_f: {}, "
        # # "left_right_rim_res: {}, "
        # # "left_right_rim_f: {}, "
        #
        # print("Best: ", "objective: {} ".format(opt.y.min()),
        #       "parameter: [left_right_rim_res: {}, "
        #       "left_right_rim_f: {}, ]".format(
        #                                      opt.X.iloc[opt.y.argmin()]['left_right_rim_res'],
        #                                      opt.X.iloc[opt.y.argmin()]['left_right_rim_f'],
        #
        #                              ))
        #

#############################################################################################################
        # f = open("/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/longrim/"+filename, 'w')
        #
        # ls = []
        # for i in range(len(rec.left_right_rim_res.values)):
        #     ls.append(rec.left_right_rim_res.values[i])
        # f.write(str(ls))
        # f.write('\n')
        #
        # ls = []
        # for i in range(len(rec.left_right_rim_f.values)):
        #     ls.append(rec.left_right_rim_f.values[i])
        # f.write(str(ls))
        # f.write('\n')
        #
        # ls = []
        # for i in range(len(obj_value)):
        #     ls.append(obj_value[i, 0])
        # f.write(str(ls))
        # f.write('\n')
        #
        # f.close()