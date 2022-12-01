import glob
import os
import numpy as np


np.random.seed(42)


broken_list = os.listdir("./data/broken")

intact_list = os.listdir("./data/intact")

ratio = [0.7, 0.2, 0.1]

lst = ["broken", "intact"]

ouput_folder_name = "data_dir"

for i, cls in enumerate([broken_list, intact_list]):

    num_data = len(cls)
    indices = list(range(num_data))
    split_valid = int(np.floor((1.0 - (ratio[1]+ratio[2])) * num_data))
    split_test = int(np.floor((1.0 - ratio[2]) * num_data))


    np.random.shuffle(indices)


    data_idx = {"train" : indices[:split_valid],
                "val":indices[split_valid:split_test],
                "test":indices[split_test:]}

    phase = ["train", "val", "test"]

    for p in phase:
        for img in data_idx[p]:
            cmd = "cp ./data/{}/{} ./{}/{}/{}/".format(lst[i],cls[img], ouput_folder_name, p, lst[i])
            os.system(cmd)
