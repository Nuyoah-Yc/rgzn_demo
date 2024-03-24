import os
import pandas as pd


def read_image_from_folder(folder_path1, folder_path2, label1, label2, save_path, path1=None, path2=None):
    img_list1 = []
    img_list2 = []
    for filename in os.listdir(folder_path1):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            if path1 is not None:
                filename = "/".join((path1, filename))
            img_list1.append(filename)
    for filename in os.listdir(folder_path2):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            if path2 is not None:
                filename = "/".join((path2, filename))
            img_list2.append(filename)
    df1 = pd.DataFrame({"path": img_list1, "label": [label1] * len(img_list1)})
    df2 = pd.DataFrame({"path": img_list2, "label": [label2] * len(img_list2)})
    df = pd.concat([df1, df2])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(save_path, header=True, index=False)
