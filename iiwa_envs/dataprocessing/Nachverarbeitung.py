import os
import numpy as np
import matplotlib.pyplot as plt
import csv


bag_dir_1 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/all_dis/"
bag_dir_2 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/"
bag_dir_3 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_ang/"
dir_list1 = os.listdir(bag_dir_1)
dir_list1.sort()
dir_list2 = os.listdir(bag_dir_2)
dir_list2.sort()
dir_list3 = os.listdir(bag_dir_3)
dir_list3.sort()

dataname1= "ipct_short_dis"
dataname2 = "ipct_long_dis"
dataname3 = "all_dis_s90"


m_objs = []
objset = []

n_suggestions = 15

ls1 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/all_dis/"
ls2 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/long/"
ls3 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/short/"
ax = plt.subplot(1, 1, 1)

for filename in dir_list1:
    with open(bag_dir_1 + filename, "r") as csvfile:
            params = []
            cr = csv.DictReader(csvfile)

            for item in cr:
                # objset.append( np.float(item["objs"].replace("[", "").replace("]", "") ) )
                # objset.append( np.float(list(item.values())[-1][-1]) )

                # params.append([int(item["iters"]), float(item["lr_params_res"]), float(item["lr_f"]),float(item["four_res"]),float(item["four_f"]),float(item["t_f"]), float(item["damp"]),float(item["objs"]) ])
                params.append([int(item["iters"]), float(item["l_res"]),float(item["l_f"]), float(item["s_res"]), float(item["s_f"]), float(item["objs"])])


            objset = np.array(objset)
            params = np.array(params)

            pre_obj = 100000000
            min_objs=[]
            mean= []
            std = []
            for i in range(int(params.shape[0] / n_suggestions )):
                iterobj = params[i*n_suggestions: (i+1)*n_suggestions, -1 ]
                mean.append(np.mean(iterobj))
                std.append(np.std(iterobj))
                if np.min(iterobj) < pre_obj:
                    pre_obj = np.min(iterobj)
                    min_objs.append(pre_obj)
                else:
                    min_objs.append(pre_obj)
            m_objs.append(min_objs)
            # Loss = []
            # mean = []
            # std = []
            # for i in range(int(len(objset) / n_suggestions) ):
            #     mean.append(np.mean(objset[n_suggestions * i: n_suggestions*i +n_suggestions] ) )
            #     std.append(np.std(objset[n_suggestions * i: n_suggestions*i +n_suggestions]))


            # mean = np.array(mean)
            # std = np.array(std)
            # iters = np.linspace(0,len(min_objs), len(min_objs) )
            # plt.plot(iters, mean)
            # plt.fill_between(iters, mean-std, mean+std, alpha=.3, label="learning rate for" + "complete traj")
            # plt.plot(np.arange(int(params.shape[0]/n_suggestions)), min_objs,label="best loss for" + "complete traj")
    with open(ls2, "r") as csvfile:
        params = []
        cr = csv.DictReader(csvfile)

        for item in cr:
            # objset.append( np.float(item["objs"].replace("[", "").replace("]", "") ) )
            # objset.append( np.float(list(item.values())[-1][-1]) )

            # params.append([int(item["iters"]), float(item["lr_params_res"]), float(item["lr_f"]),float(item["four_res"]),float(item["four_f"]),float(item["t_f"]), float(item["damp"]),float(item["objs"]) ])
            params.append(
                [int(item["iters"]), float(item["res"]), float(item["f"]), float(item["t_f"]), float(item["damp"]), float(item["objs"])])

        objset = np.array(objset)
        params = np.array(params)

        pre_obj = 100000000
        min_objs = []
        mean = []
        std = []
        for i in range(int(params.shape[0] / n_suggestions)):
            iterobj = params[i * n_suggestions: (i + 1) * n_suggestions, -1]
            mean.append(np.mean(iterobj))
            std.append(np.std(iterobj))
            if np.min(iterobj) < pre_obj:
                pre_obj = np.min(iterobj)
                min_objs.append(pre_obj)
            else:
                min_objs.append(pre_obj)
        m_objs.append(min_objs)
        # Loss = []
        # mean = []
        # std = []
        # for i in range(int(len(objset) / n_suggestions) ):
        #     mean.append(np.mean(objset[n_suggestions * i: n_suggestions*i +n_suggestions] ) )
        #     std.append(np.std(objset[n_suggestions * i: n_suggestions*i +n_suggestions]))

        # mean = np.array(mean)
        # std = np.array(std)
        # iters = np.linspace(0,len(min_objs), len(min_objs) )
        # plt.plot(iters, mean)
        # plt.fill_between(iters, mean-std, mean+std, alpha=.3, label="learning rate for" + "complete traj")
        # plt.plot(np.arange(int(params.shape[0]/n_suggestions)), min_objs,label="best loss for" + "complete traj")
    with open(ls3, "r") as csvfile:
        params = []
        cr = csv.DictReader(csvfile)

        for item in cr:
            params.append( [int(item["iters"]), float(item["res"]), float(item["f"]), float(item["objs"])])

        objset = np.array(objset)
        params = np.array(params)

        pre_obj = 100000000
        min_objs = []
        mean = []
        std = []
        for i in range(int(params.shape[0] / n_suggestions)):
            iterobj = params[i * n_suggestions: (i + 1) * n_suggestions, -1]
            mean.append(np.mean(iterobj))
            std.append(np.std(iterobj))
            if np.min(iterobj) < pre_obj:
                pre_obj = np.min(iterobj)
                min_objs.append(pre_obj)
            else:
                min_objs.append(pre_obj)
        m_objs.append(min_objs)


fig, ax = plt.subplots(figsize=(7, 4))

# standard error bars
ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)
# for min_objs in m_objs:
#     # mean = np.array(mean)
#     # std = np.array(std)
#     iters = np.linspace(0,len(min_objs), len(min_objs) )
#
#     plt.plot(np.arange(int(params.shape[0]/n_suggestions)), min_objs,label="best loss for" + "complete traj")
#     print("best params are",[float(p) for p in params[np.argmin(params[:,-1]), :]])


plt.legend()
plt.show()


# plt.plot(np.arange(objs.shape[0]), np.mean(objs, axis=1))
# plt.fill_between(np.arange(objs.shape[0]), np.mean(objs, axis=1) - np.std(objs, axis=1), np.mean(objs, axis=1) + np.std(objs, axis=1), alpha=0.3)
# plt.show()
None
