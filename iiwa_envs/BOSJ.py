import pandas as pd
import numpy as np
from bo.design_space.design_space import DesignSpace
from bo.optimizers.hebo import HEBO
from sim_With_bullet import *

def obj(params) -> np.ndarray:
    out = np.zeros((params.shape[0], 1))
    for i in range(params.shape[0]):
        X = [params.iloc[i].get('t_lateral_f'),
             params.iloc[i].get('left_right_rim_res'),
             params.iloc[i].get('left_right_rim_f'),
             params.iloc[i].get('four_side_rim_res'),
             params.iloc[i].get('four_side_rim_latf'),
             params.iloc[i].get('angvel'),

             ]
        model = Model(X, init_pos.copy(), init_vel.copy())
        t_sim, sim_pos, _ = model.sim_bullet('DIRECT')
        simdata = np.hstack((t_sim, sim_pos))
        # loss = Lossfun(bagdata.copy(), simdata, 'DIRECT')
        loss = get_Err(bagdata.copy(), simdata.copy())

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

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()

    # choose bag file
    bag_name = dir_list[7]
    bag = rosbag.Bag(os.path.join(bag_dir, bag_name))
    print(bag_name)
    bagdata = read_bag(bag)  # [n,7]

    # get linear velocity

    lin_ang_vel = get_vel(bagdata.copy())  # [n,6]
    init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]
    #  get init vel + vel at z direction
    lin_vel, ang_vel = vel2initvel(lin_ang_vel, bagdata.copy())
    init_vel = np.hstack((np.hstack((lin_vel, 0)), ang_vel))

#  [t_lateral_f, left_right_rim_res, left_rim_f, four_side_rim_res, four_side_rim_latf, angvel]  [5,]
    param_list = [
        {'name': 't_lateral_f', 'type': 'num', 'lb': 0., 'ub': .1},
        {'name': 'left_right_rim_res', 'type': 'num', 'lb': .5, 'ub': .9},
        {'name': 'left_right_rim_f', 'type': 'num', 'lb': 0., 'ub': .5},
        {'name': 'four_side_rim_res', 'type': 'num', 'lb': .5, 'ub': .9},
        {'name': 'four_side_rim_latf', 'type': 'num', 'lb': 0., 'ub': .5}

    ]

    space = DesignSpace().parse(param_list)
    opt = HEBO(space)

    # init value
    rec = pd.DataFrame({
                        't_lateral_f': [.1, .01, .004, .3, .08, .03,],
                        'left_right_rim_res': [.9, .7, .5, .4, .67, .75] ,
                        'left_right_rim_f': [.1, .3, .5, .2, .25, .4],
                        'four_side_rim_res': [.9, .7, .5, .4, .67, .75],
                        'four_side_rim_latf':[.1, .3, .5, .2, .25, .4]


                        }
                        )


    objs = []
    params = []
    for i in range(1000):
        rec = opt.suggest(n_suggestions=4)
        opt.observe(rec, obj(rec))
        obj_value = obj(rec)
        print('iter', i)
        params.append([rec.t_lateral_f.values, rec.left_right_rim_res.values, rec.left_right_rim_f.values, rec.four_side_rim_res.values,
                       rec.four_side_rim_latf.values, rec.angvel.values])
        objs.append(obj_value)
        print("Current: ", "objective: {} ".format(obj_value[0, 0]),
              "parameter: [t_lateral_f: {}, "
              "left_right_rim_res: {}, "
              "left_right_rim_f: {}, "
              "four_side_rim_res: {}, "
              "four_side_rim_latf: {},]".format(rec.t_lateral_f.values[0], rec.left_right_rim_res.values[0], rec.left_right_rim_f.values[0], rec.four_side_rim_res.values[0],
                       rec.four_side_rim_latf.values[0], rec.angvel.values[0]))
        print("Best: ", "objective: {} ".format(opt.y.min()),
              "parameter: [t_lateral_f: {}, "
              "left_right_rim_res: {}, "
              "left_right_rim_f: {}, "
              "four_side_rim_res: {}, "
              "four_side_rim_latf: {}, ]".format(opt.X.iloc[opt.y.argmin()]['t_lateral_f'],
                                             opt.X.iloc[opt.y.argmin()]['left_right_rim_res'],
                                             opt.X.iloc[opt.y.argmin()]['left_right_rim_f'],
                                             opt.X.iloc[opt.y.argmin()]['four_side_rim_res'],
                                             opt.X.iloc[opt.y.argmin()]['four_side_rim_latf'],

                                     ))

    f = open('i50n10_trajloss.txt', 'w')
    #
    f.write('*' * 50)
    f.write('parameters')
    f.write('*' * 50)
    f.write('\n')
    for x in params:
        f.write(str(x))
        f.write('\n')
    f.write('*' * 50)
    f.write('Loss')
    f.write('*' * 50)
    f.write('\n')
    for loss in objs:
        f.write(str(loss))
        f.write('\n')


    f.close()