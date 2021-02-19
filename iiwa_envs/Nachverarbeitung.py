import os
import numpy as np
import matplotlib.pyplot as plt


bag_dir_1 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/shortrim/"
bag_dir_2 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/longrim/"


dirlist_1 = os.listdir(bag_dir_1)
dirlist_1.sort()
dirlist_2 = os.listdir(bag_dir_2)
dirlist_2.sort()


datas = []
objs = []

Params_res = []
Params_f =[]
for filename in dirlist_1:
    with open(bag_dir_1 + filename, "r") as reader:
        data = []
        obj = []
        params_res = []
        params_f = []

        for line in reader:
            line = line.replace("[", " ").replace("]", " ").replace(",", " ").replace("\n", " ").split()
            data.append(np.float64(line))
        data = np.array(data)
        datas.append(data)
        for i in range(2, data.shape[0], 3):
            obj.append(data[i])
        for i in range(0, data.shape[0], 3):
            params_res.append(data[i])
        for i in range(1, data.shape[0], 3):
            params_f.append(data[i])

        obj = np.array(obj)
        params_res = np.array(params_res)
        params_f = np.array(params_f)

        objs.append(obj)
        Params_res.append(params_res)
        Params_f.append(params_f)
datas = np.array(datas)
objs = np.array(objs)
objs = np.hstack([objs[i,:,:] for i in range(objs.shape[0])])
Params_res = np.array(Params_res)
Params_res = np.hstack([Params_res[i,:,:] for i in range(Params_res.shape[0])])

Params_f = np.array(Params_f)
Params_f = np.hstack([Params_f[i,:,:] for i in range(Params_f.shape[0])])

plt.plot(np.arange(objs.shape[0]), np.mean(objs, axis=1))
plt.fill_between(np.arange(objs.shape[0]), np.mean(objs, axis=1) - np.std(objs, axis=1), np.mean(objs, axis=1) + np.std(objs, axis=1), alpha=0.3)
plt.show()
None
