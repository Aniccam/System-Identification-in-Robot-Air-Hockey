import os
import numpy as np
import matplotlib.pyplot as plt
import csv


bag_dir_1 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/all_dis/"
bag_dir_2 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/long/"
bag_dir_3 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/short/"
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
y=[]
xerr = []
yerr =[]
n_suggestions = 15
best=[]
ls1 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/all_dis/"
ls2 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/long/"
ls3 = "/home/hszlyw/PycharmProjects/ahky/iiwa_envs/results_v1/zum Ausarbeitung2/seg_dis/short/"
ax = plt.subplot(1, 1, 1)
Params = []
best = []
for filename in dir_list1:
    params = []
    with open(bag_dir_1 + filename, "r") as csvfile:

        cr = csv.DictReader(csvfile)

        for item in cr:
            params.append(float(item["objs"]))
            Params.append(float(item["objs"]))
        best.append(np.min(params))
y.append(np.mean(best))
yerr.append(2* np.std(Params))
xerr.append(2* np.std(best))

Params = []
best = []

for filename in dir_list2:
    params = []
    with open(bag_dir_2 + filename, "r") as csvfile:

        cr = csv.DictReader(csvfile)

        for item in cr:
            params.append(float(item["objs"]))
            Params.append(float(item["objs"]))
        best.append(np.min(params))
y.append(np.mean(best))
yerr.append(2* np.std(Params))
xerr.append(2* np.std(best))

Params = []
best = []

for filename in dir_list3:
    params = []
    with open(bag_dir_3 + filename, "r") as csvfile:

        cr = csv.DictReader(csvfile)

        for item in cr:
            params.append(float(item["objs"]))
            Params.append(float(item["objs"]))
        best.append(np.min(params))
y.append(np.mean(best))
yerr.append(2* np.std(Params))
xerr.append(2* np.std(best))



fig, ax = plt.subplots()
# standard error bars
# ax.axis('equal')
ax.ylim=(0, .29)
ax.errorbar(0.1, y[0], xerr=xerr[0], yerr=yerr[0],  marker='d', markersize=5, color="tab:orange")
ax.errorbar(0.18, y[1], xerr=xerr[1], yerr=yerr[1],  marker='d', markersize=5, color="tab:green")
ax.errorbar(.26, y[2], xerr=xerr[2], yerr=yerr[2],  marker='d', markersize=5, color="tab:purple")
ax.set_xticks([0.1, .18, .26])
ax.set_xticklabels(['overall movement', 'long rim collision', 'short rim collision'], fontsize=10)
ax.set_ylabel('loss', fontsize=13)

# plt.legend()
plt.show()


# plt.plot(np.arange(objs.shape[0]), np.mean(objs, axis=1))
# plt.fill_between(np.arange(objs.shape[0]), np.mean(objs, axis=1) - np.std(objs, axis=1), np.mean(objs, axis=1) + np.std(objs, axis=1), alpha=0.3)
# plt.show()
None
