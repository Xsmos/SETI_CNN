import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import h5py


def preprocess(dataset, debug=False):
    """
    This function substracts averaged OFF-cadences from ON-cadences.
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
            all_figures_in_dataset = np.vstack(
                (all_figures_in_dataset, on_substracted.reshape(1, on_substracted.shape[0], on_substracted.shape[1], on_substracted.shape[2])))
            labels_in_dataset = np.append(
                labels_in_dataset, int(train_labels.target.iloc[i]))
            # ids_of_labels = np.append(ids_of_labels, idx)
            print(
                "{}/{}".format(all_figures_in_dataset.shape[0], train_labels.id.shape[0]), end="\r")
        else:
            print("Initialize the variables...")
            all_figures_in_dataset = np.array([on_substracted])
            labels_in_dataset = int(train_labels.target.iloc[i])
            # ids_of_labels = idx

        # sleep(10)
        if debug and all_figures_in_dataset.shape[0] >= 100:
            break

    if not os.path.exists(os.path.join(inputDir, "train_preprocessed")):
        os.makedirs(os.path.join(inputDir, "train_preprocessed"))

    with h5py.File(os.path.join(inputDir, 'train_preprocessed', "{}.h5".format(dataset)), 'w') as hf:
        # print(ids_of_labels)
        # hf.create_dataset("id", data = ids_of_labels)
        print("\ntarget.shape = ", labels_in_dataset.shape)
        hf.create_dataset("target", data=labels_in_dataset)
        print("figure.shape = ", all_figures_in_dataset.shape)
        hf.create_dataset("figure", data=all_figures_in_dataset)

    t_end = time()
    total_time = t_end - t_start
    print("It costs {:.3f} s to preprocess the dataset.".format((total_time)))


if __name__ == "__main__":
    preprocess(1)
