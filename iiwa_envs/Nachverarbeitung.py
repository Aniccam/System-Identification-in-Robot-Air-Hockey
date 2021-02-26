import os
import numpy as np
import matplotlib.pyplot as plt
import csv


bag_dir_1 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/"
bag_dir_2 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/longrim/"

objset = []
params = []
n_suggestions = 15
with open(bag_dir_1 + "combloss_short_n15_iter70", "r") as csvfile:
    cr = csv.DictReader(csvfile)

    for item in cr:
        # objset.append( np.float(item["objs"].replace("[", "").replace("]", "") ) )
        objset.append( np.float(item["objs"]) )

        params.append([item["params_res"], item["params_f"]])

    # objset = np.array(objset)
    params = np.array(params)
    Loss = []
    mean = []
    std = []
    for i in range(int(len(objset) / n_suggestions) ):
        mean.append(np.mean(objset[n_suggestions * i: n_suggestions*i +n_suggestions] ) )
        std.append(np.std(objset[n_suggestions * i: n_suggestions*i +n_suggestions]))


    mean = np.array(mean)
    std = np.array(std)
    iters = np.linspace(0,int(len(objset) / n_suggestions), int(len(objset)/ n_suggestions) )
    plt.plot(iters, mean)
    plt.fill_between(iters, mean-std, mean+std, alpha=.3, label="short rim")
    plt.legend()
    plt.show()
    print("best params are",params[np.argmin(objset[:, 1])])




# plt.plot(np.arange(objs.shape[0]), np.mean(objs, axis=1))
# plt.fill_between(np.arange(objs.shape[0]), np.mean(objs, axis=1) - np.std(objs, axis=1), np.mean(objs, axis=1) + np.std(objs, axis=1), alpha=0.3)
# plt.show()
# None
