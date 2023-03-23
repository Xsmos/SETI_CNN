import os
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

def preprocess(dataset):
    """
    reshape and normalize the raw data into files .npy, labels.npy, and ids.npy.
    input: [0,9] U [a,f]
    """
    dataset = str(dataset)
    file_names = os.listdir("./train/"+dataset)
    print("{} files in ./train/{}/ are going to be preprocessed.".format(len(file_names), dataset))
    print("It's a 3GB file so make sure your machine has adequate memory; It needs around 20 minutes to preprocess.")
    
    train_labels_array = np.loadtxt('train_labels.csv', delimiter=',', dtype=str)
    train_labels_dict = {train_labels_array[i,0]: train_labels_array[i,1] for i in range(train_labels_array.shape[0])}
    
    for file_name in file_names:
        # print(file_name)
        
        # merge six figure into one
        data_raw = np.load("./train/"+dataset+"/"+file_name)
        data_1D = data_raw.reshape(np.prod(data_raw.shape))
        data_2D = data_1D.reshape(data_raw.shape[1]*6, data_raw.shape[2])
        # print("raw data shape =", data_raw.shape)
        # # print(data_1D.shape)
        # print("present shape =", data_2D.shape)
        
        # normalize the merged figure data
        data_normalized = (data_2D-data_2D.min())/(data_2D.max()-data_2D.min())
        # print("min =", data_normalized.min())
        # print("max =", data_normalized.max())
        
        # check the data by plotting the figure
        # plt.figure(figsize=(6,18))
        # plt.imshow(data_normalized)
        # plt.show()
        
        if "all_figures_in_dataset" in vars():
            all_figures_in_dataset = np.vstack((all_figures_in_dataset, [data_normalized]))
            labels_in_dataset = np.append(labels_in_dataset, int(train_labels_dict[file_name[:-4]]))
            # print("labels_in_dataset =", labels_in_dataset)
            ids_of_labels = np.append(ids_of_labels, file_name[:-4])
            
            percent = int(all_figures_in_dataset.shape[0]/len(file_names) * 100)
            if int(percent) % 5 == 0:
                print("{}%".format(percent), end="\r")
        else:
            print("Initialize the variables...")
            all_figures_in_dataset = np.array([data_normalized])
            labels_in_dataset = int(train_labels_dict[file_name[:-4]])
            # print("labels_in_dataset =", labels_in_dataset)
            ids_of_labels = file_name[:-4]
        # print("shape of all_figures_in_dataset =", all_figures_in_dataset.shape)
        
        # sleep(10)
        # if all_figures_in_dataset.shape[0] >= 100:
        #     break
        
    print("\nshape of all_figures_in_dataset =", all_figures_in_dataset.shape)
    
    if not os.path.exists("./train_preprocessed/"):
        os.makedirs("./train_preprocessed/")
        
    np.save("./train_preprocessed/"+dataset, all_figures_in_dataset)
    np.save("./train_preprocessed/"+dataset+"_labels", labels_in_dataset)
    np.save("./train_preprocessed/"+dataset+"_ids", ids_of_labels)

if __name__ == "__main__":
    preprocess(1)
