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
             params.iloc[i].get('left_right_rim_f')


             ]
        #,params.iloc[i].get('latf'),
        model = Model(X, init_pos.copy(), init_vel.copy())
        t_sim, sim_pos, _ = model.sim_bullet(table, 'DIRECT')
        simdata = np.hstack((t_sim, sim_pos))
        # loss = Lossfun(bagdata.copy(), simdata, 'DIRECT')
        loss = Lossfun2(bagdata.copy(), simdata.copy(), table, 'DIRECT')
        # plt.plot(simdata[:,1], simdata[:,2], label='sim')
        # plt.plot(bagdata[:,1], bagdata[:,2], label='bag')
        # plt.legend()
        # plt.show()


        out[i] = loss
    return out



if __name__ == "__main__":

    bag_dir = "/home/hszlyw/Documents/airhockey/rosbag/edited/dataset_short/"

    dir_list = os.listdir(bag_dir)
    dir_list.sort()


#  [t_lateral_f, left_right_rim_res, left_rim_f, four_side_rim_res, four_side_rim_latf, angvel]  [5,]
    param_list = [
        {'name': 'left_right_rim_res', 'type': 'num', 'lb': .5, 'ub': .93},
        {'name': 'left_right_rim_f', 'type': 'num', 'lb': 0., 'ub': .8}

    ]
  #{'name': 'latf', 'type': 'num', 'lb': .01, 'ub': .09}
    space = DesignSpace().parse(param_list)
    opt = HEBO(space)

    # init value
    rec = pd.DataFrame({

                        'left_right_rim_res': [.9, .7, .95, .59, .67, .75, .62, .8],
                        'left_right_rim_f': [.1, .3, .45, .6, .01, .4, .6, .7]


                        }
                        )


    #,'latf': [.04, .06, .07, .08, .06, .065, .09, .021]
    # filename = "24.txt"
    # print(filename)
    data = []

    objs = []
    params = []
    n_sugg = 15
    fileHeader = ["iters", "params_res", "params_f", "objs" ]
    with open("/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/" + "combloss_short_n15_iter70", 'w') as csvfile:
        cw = csv.writer(csvfile)
        cw.writerow(fileHeader)



        for iter in range(70):
            rec = opt.suggest(n_suggestions=n_sugg)
            # obj_value = np.zeros((n_sugg, 1)) # nrows = n_suggestions
            obj_value = []
            mean_obj = []

            for filename in dir_list:
                # filename = dir_list[1]
                # filename = "111.txt"

                # print(filename)

                data = []
                with open(bag_dir + filename, 'r') as f:
                    for line in f:
                        data.append(np.array(np.float64(line.replace("[", " ").replace("]", " ").replace(",", " ").split())))
                    bagdata = np.array(data)
                    table = bagdata[0, :]
                    table[2] = 0.11945
                    bagdata = bagdata[1:, :]
                    bagdata[:, 3] = 0.11945 * np.ones(bagdata.shape[0])
                    init_pos = np.hstack((bagdata.copy()[0, 1:3], 0.11945))  # [3,]
                    lin_ang_vel = get_vel(bagdata.copy())  # In [n,7], Out[n,6]
                    begin_idx = np.argmax(np.abs(lin_ang_vel[:, 2]))      ######### see vel_y [:,1] if hit shortrim
                    init_vel = lin_ang_vel[begin_idx, :]

                    # bagdata = bagdata[:back_process(lin_ang_vel), :]

                    tempobj = obj(rec, table)
                    # while (tempobj == 180.3334).any():  # 180.3334 wrong loss, should remove
                    #     tempobj = np.delete(tempobj, np.where(tempobj == 180.3334))

                    # tempobj = np.mean(tempobj)

                    # obj_value = tempobj + obj_value       # float64
                    obj_value.append(tempobj)

            obj_value = np.array(obj_value).squeeze().T
            for i, objs in enumerate(obj_value):
                mean_obj.append(np.mean([obj for obj in objs if obj != 180.3334 ] ) * 100 )

            mean_obj = np.array(mean_obj).reshape(n_sugg,1)
            for j in range(n_sugg):


                cw.writerow([
                    str(iter),
                    str(rec.left_right_rim_res.values[j]),
                    str(rec.left_right_rim_f.values[j]),
                    str(np.float(mean_obj[j]))])

            # print(obj_value)
            opt.observe(rec, mean_obj)
            # objs.append(obj_value)


            print('iter', iter)
            #
            print("Current: ", "objective: {} ".format(mean_obj[0, 0]),
                  "parameter: [left_right_rim_res: {}, "
                  "left_right_rim_latf: {}," .format(rec.left_right_rim_res.values[0],
                           rec.left_right_rim_f.values[0]) )

#                      "lateral_friction: {} ]"
            #                            rec.latf.values[0]

            print("Best: ", "objective: {} ".format(opt.y.min()),
                  "parameter: [left_right_rim_res: {}, "
                  "left_right_rim_f: {}, ".format(
                                                 opt.X.iloc[opt.y.argmin()]['left_right_rim_res'],
                                                 opt.X.iloc[opt.y.argmin()]['left_right_rim_f'],


                                         ))

            #
            #      "lateral_friction: {} ]"     opt.X.iloc[opt.y.argmin()]['latf']