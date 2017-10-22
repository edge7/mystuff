import os
import pathlib

import subprocess

if __name__ == "__main__":
    input_dir = "/home/edge7/Desktop/MLApplied/Forex/Data/Stock"
    flist = [p for p in pathlib.Path(input_dir).iterdir() if p.is_dir()]
    train_len_list = reversed(range(20, 340, 50))
    for train_len in train_len_list:
        for stock_dir in flist:
            print("Running: " + str(stock_dir) + " con train_len: " + str(train_len))
            command = "python3.5 /home/edge7/PycharmProjects/MlStock/ingestion/main.py --datapath "
            command += str(stock_dir)
            target = str(stock_dir).split("/")[-1]
            command += " --target " + target
            command += " --train_len " + str(train_len)
            with open(os.devnull, 'w') as devnull:
                subprocess.run(command, check=True, shell=True, stdout=devnull, stderr=devnull)