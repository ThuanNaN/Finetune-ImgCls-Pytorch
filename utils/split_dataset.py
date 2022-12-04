
import os
import glob
import numpy as np
import concurrent.futures

def kfold_split(input_dir, output_dir, classes, seed, idx):

    for cls in classes:

        image_list = list(img_name for img_name in glob.glob("{}/{}/*.jpg".format(input_dir, cls)))


        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        fold_dir = os.path.join(output_dir,"fold_{}".format(idx))
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)


        np.random.seed(seed)


        num_data = len(image_list)
        indices = list(range(num_data))
        split_valid = int(np.floor((1.0 - (ratio[1]+ratio[2])) * num_data))
        split_test = int(np.floor((1.0 - ratio[2]) * num_data))
        np.random.shuffle(indices)

        data_idx = {
            "train" : indices[:split_valid],
            "val":indices[split_valid:split_test],
            "test":indices[split_test:]
        }

        for p in ["train", "val", "test"]:
            phase_path = os.path.join(fold_dir, p)
            if not os.path.exists(phase_path):
                os.mkdir(phase_path)

            role_path = os.path.join(phase_path, cls)
            if not os.path.exists(role_path):
                os.mkdir(role_path)

            
            for idx_img in data_idx[p]:
                cmd = "cp {} {}".format(image_list[idx_img], role_path)
                os.system(cmd)

        
ratio = [0.7, 0.2, 0.1]

if __name__ == "__main__":

    seed_arr = [11, 13, 17, 21, 42]
    classes = ["broken", "intact"]
    input_dir = "./data/data_trainAug"
    ouput_dir = "./data/KFold_data"
    # with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #     for idx, seed in enumerate(seed_arr):
    #         executor.submit(kfold_split, input_dir, ouput_dir, classes, seed, idx)
    for idx, seed in enumerate(seed_arr):
        kfold_split(input_dir, ouput_dir, classes, seed, idx)

