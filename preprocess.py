import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import h5py

dt = h5py.special_dtype(vlen=str)

def subtract_both(dataset, which="test", debug=False):
    dataset = str(dataset)
    inputDir = "./seti-breakthrough-listen"
    dataDir = os.path.join(inputDir, which, dataset)

    t_start = time()
    id_list = os.listdir(dataDir)

    for idx in id_list:
        x = np.load(os.path.join(dataDir, idx))
        off_averaged = np.average(x[1::2], axis=0)
        on_substracted = x[0::2] - off_averaged

        if "all_figures_in_dataset" in vars():
            all_figures_in_dataset = np.vstack((all_figures_in_dataset, on_substracted.reshape(
                1, on_substracted.shape[0], on_substracted.shape[1], on_substracted.shape[2])))
            # labels_in_dataset = np.append(labels_in_dataset, int(train_labels.target.iloc[i]))
            ids_of_labels = np.append(ids_of_labels, idx[:-4])
            print("{}/{}".format(all_figures_in_dataset.shape[0], len(id_list)), end="\r")
        else:
            print("Preprocessing {}/...".format(dataset))
            all_figures_in_dataset = np.array([on_substracted])
            # labels_in_dataset = int(train_labels.target.iloc[i])
            ids_of_labels = idx[:-4]

        if debug and all_figures_in_dataset.shape[0] >= 100:
            break

    if not os.path.exists(os.path.join(inputDir, which+"_preprocessed")):
        os.makedirs(os.path.join(inputDir, which+"_preprocessed"))

    with h5py.File(os.path.join(inputDir, which+'_preprocessed', "{}.h5".format(dataset)), 'w') as hf:
        # print("\ntarget.shape = ", labels_in_dataset.shape)
        # hf.create_dataset("target", data=labels_in_dataset)
        print("figure.shape = ", all_figures_in_dataset.shape)
        hf.create_dataset("figure", data=all_figures_in_dataset)
        print("id.shape =", ids_of_labels.shape)
        hf.create_dataset("id", dtype=dt, data=ids_of_labels.tolist())

    t_end = time()
    total_time = t_end - t_start
    print("It costs {:.3f} s to preprocess the dataset.".format((total_time)))


def subtract(dataset, debug=False):
    """
    This function subtract averaged OFF-cadences from ON-cadences.
    input: [0,9] U [a,f]
    output: .h5 consisting of 'figure' and 'target'
    """
    dataset = str(dataset)

    inputDir = "./seti-breakthrough-listen"

    train_labels = pd.read_csv(os.path.join(inputDir, 'train_labels.csv'))
    True_False = [idx[0] == dataset for idx in train_labels.id]
    train_labels = train_labels[True_False]

    t_start = time()
    for i, idx in enumerate(train_labels.id):
        x = np.load(os.path.join(inputDir, "train", idx[0], idx+'.npy'))

        off_averaged = np.average(x[1::2], axis=0)
        on_substracted = x[0::2] - off_averaged

        if "all_figures_in_dataset" in vars():
            all_figures_in_dataset = np.vstack((all_figures_in_dataset, on_substracted.reshape(
                1, on_substracted.shape[0], on_substracted.shape[1], on_substracted.shape[2])))
            labels_in_dataset = np.append(
                labels_in_dataset, int(train_labels.target.iloc[i]))
            ids_of_labels = np.append(ids_of_labels, idx)
            print(
                "{}/{}".format(all_figures_in_dataset.shape[0], train_labels.id.shape[0]), end="\r")
        else:
            print("Preprocessing {}/...".format(dataset))
            all_figures_in_dataset = np.array([on_substracted])
            labels_in_dataset = int(train_labels.target.iloc[i])
            ids_of_labels = idx

        if debug and all_figures_in_dataset.shape[0] >= 100:
            break

    if not os.path.exists(os.path.join(inputDir, "train_preprocessed")):
        os.makedirs(os.path.join(inputDir, "train_preprocessed"))

    with h5py.File(os.path.join(inputDir, 'train_preprocessed', "{}.h5".format(dataset)), 'w') as hf:
        print("\ntarget.shape = ", labels_in_dataset.shape)
        hf.create_dataset("target", data=labels_in_dataset)
        print("figure.shape = ", all_figures_in_dataset.shape)
        hf.create_dataset("figure", data=all_figures_in_dataset)
        print("id.shape =", ids_of_labels.shape)
        hf.create_dataset("id", dtype=dt, data=ids_of_labels.tolist())

    t_end = time()
    total_time = t_end - t_start
    print("It costs {:.3f} s to preprocess the dataset.".format((total_time)))


def combine(which="test", all=False, debug=False):
    """
    Combine every subtracted .h5 files in which into single one .h5 file.
    If all=True, include files that were not downloaded yet and subtract them to get corresponding .h5s.
    """

    dir = os.path.join("./seti-breakthrough-listen", which+"_preprocessed")

    if all:
        file_list = np.append(np.arange(10).astype(
            str), ['a', 'b', 'c', 'd', 'e', 'f'])
    else:
        file_list = np.array([0, 1], dtype=str)

    for i in file_list:
        if os.path.exists(os.path.join(dir, i+".h5")):
            print("{}.h5 exists, skipping.".format(i))
            continue
        else:
            if which == 'train':
                subtract(i, debug=debug)
            elif which == "test":
                subtract_both(i, which, debug)
            else:
                print("'which' should be one of 'train' or 'test'.")
            print("---"*30)

    save_dir = os.path.join(dir, which+".h5")
    if os.path.exists(save_dir):
        print(which+".h5 already exists, skipping.")
    else:
        with h5py.File(save_dir, 'a') as file:
            for name in file_list:
                print("Combining file {}.h5 ...".format(name))
                with h5py.File(os.path.join(dir, name+'.h5'), 'r') as f:
                    # file.create_group(name)
                    if name == '0':
                        figure = file.create_dataset(
                            'figure', data=f['figure'], maxshape=(None,None,None,None))
                        idx = file.create_dataset(
                            "id", data=f['id'], maxshape=(None, ))
                        if which == 'train':
                            target = file.create_dataset("target", data=f['target'], maxshape=(None,))
                    else:
                        figure.resize(
                            figure.shape[0]+f['figure'].shape[0], axis=0)
                        figure[-f['figure'].shape[0]:] = f['figure']
                        idx.resize(idx.shape[0]+f['id'].shape[0], axis=0)
                        idx[-f['id'].shape[0]:] = f['id']
                        if which == "train":
                            target.resize(target.shape[0]+f['target'].shape[0], axis=0)
                            target[-f['target'].shape[0]:] = f['target']


if __name__ == "__main__":
    # subtract(1)
    combine("test", all=True, debug=False)
