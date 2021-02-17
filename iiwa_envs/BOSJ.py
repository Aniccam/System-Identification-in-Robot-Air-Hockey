import pandas as pd
import numpy as np
from bo.design_space.design_space import DesignSpace
from bo.optimizers.hebo import HEBO
from sim_With_bullet import *

def obj(params) -> np.ndarray:
    out = np.zeros((params.shape[0], 1))
    for i in range(params.shape[0]):
        X = [
             params.iloc[i].get('four_side_rim_res'),
             params.iloc[i].get('four_side_rim_latf'),


             ]
        model = Model(X, init_pos.copy(), init_vel.copy())
        t_sim, sim_pos, _ = model.sim_bullet('DIRECT')
        simdata = np.hstack((t_sim, sim_pos))
        # loss = Lossfun(bagdata.copy(), simdata, 'DIRECT')
        loss = Lossfun2(bagdata.copy(), simdata.copy(), 'DIRECT')

        out[i] = loss
    return out
    #
    # X = [param.get('restitution'),  param.get('lateral_friction_siderim') ]
    # model = Model(X, init_pos.copy(), init_vel.copy())
    # t_sim, sim_pos, _ = model.sim_bullet('DIRECT')
    # simdata = np.hstack((t_sim, sim_pos))
    # loss = Lossfun(bagdata.copy(), simdata)
    # return loss



if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited/shortrim_collision/"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()

    for filename in dir_list:
        #
        # filename = dir_list[4]
        print(filename)
        data = []

        with open(bag_dir + filename, 'r') as f:
            for line in f:
                data.append(np.array(np.float64(line.replace("[", " ").replace("]", " ").replace(",", " ").split())))
            bagdata = np.array(data)

        lin_ang_vel = get_vel(bagdata.copy())  # In [n,7], Out[n,6]

        bagdata = bagdata[:back_process(lin_ang_vel), :]
        init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]
        #  get init vel + vel at z direction
        lin_vel, ang_vel = vel2initvel(lin_ang_vel, bagdata.copy())
        init_vel = np.hstack((np.hstack((lin_vel, 0)), ang_vel))

    #  [t_lateral_f, left_right_rim_res, left_rim_f, four_side_rim_res, four_side_rim_latf, angvel]  [5,]
        param_list = [
            {'name': 'four_side_rim_res', 'type': 'num', 'lb': .4, 'ub': .9},
            {'name': 'four_side_rim_latf', 'type': 'num', 'lb': 0., 'ub': .7}

        ]
        # {'name': 't_lateral_f', 'type': 'num', 'lb': 0., 'ub': .1},
        # {'name': 'left_right_rim_res', 'type': 'num', 'lb': .5, 'ub': .9},
        # {'name': 'left_right_rim_f', 'type': 'num', 'lb': 0., 'ub': .5},
        space = DesignSpace().parse(param_list)
        opt = HEBO(space)

        # init value
        rec = pd.DataFrame({

                            'four_side_rim_res': [.9, .7, .5, .4, .67, .75],
                            'four_side_rim_latf':[.1, .3, .5, .2, .01, .4]


                            }
                            )
        # 't_lateral_f': [.1, .01, .004, .3, .08, .03, ],
        # 'left_right_rim_res': [.9, .7, .5, .4, .67, .75],
        # 'left_right_rim_f': [.1, .3, .5, .2, .25, .4],

        objs = []
        params = []
        f = open("/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/"+filename, 'w')

        for i in range(50):
            rec = opt.suggest(n_suggestions=4)
            opt.observe(rec, obj(rec))
            obj_value = obj(rec)
            print('iter', i)

            ls= []
            for i in range(len(rec.four_side_rim_res.values)):
                ls.append(rec.four_side_rim_res.values[i])
            f.write(str(ls))
            f.write('\n')

            ls = []
            for i in range(len(rec.four_side_rim_latf.values)):
                ls.append(rec.four_side_rim_latf.values[i])
            f.write(str(ls))
            f.write('\n')

            ls=[]
            for i in range(len(obj_value)):
                ls.append(obj_value[i,0])
            f.write(str(ls))
            f.write('\n')


            print("Current: ", "objective: {} ".format(obj_value[0, 0]),
                  "parameter: [four_side_rim_res: {}, "
                  "four_side_rim_latf: {},]".format(rec.four_side_rim_res.values[0],
                           rec.four_side_rim_latf.values[0]))
            # rec.t_lateral_f.values[0], rec.left_right_rim_res.values[0], rec.left_right_rim_f.values[0],
            # t_lateral_f: {}, "
            # "left_right_rim_res: {}, "
            # "left_right_rim_f: {}, "

            print("Best: ", "objective: {} ".format(opt.y.min()),
                  "parameter: [four_side_rim_res: {}, "
                  "four_side_rim_latf: {}, ]".format(
                                                 opt.X.iloc[opt.y.argmin()]['four_side_rim_res'],
                                                 opt.X.iloc[opt.y.argmin()]['four_side_rim_latf'],

                                         ))



        f.close()