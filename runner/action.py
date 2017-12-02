import os
import pathlib #e.durso@live.com
import subprocess

input_dir = "/home/toniotonia47/Desktop/stockML/data/"
# input_dir = "/home/edge7/Desktop/MLApplied/Forex/Data/Stock/"
best_lens = {input_dir + "CADJPY": [400,450,500,550,600,700],
             }

if __name__ == "__main__":

    flist = [p for p in pathlib.Path(input_dir).iterdir() if p.is_dir()]
    train_len_list = reversed(range(20, 550, 50))
    for stock_dir in flist:
        train_lens = best_lens.get(str(stock_dir), None)
        if train_lens is None:
            print("*** Unable to run " + str(stock_dir) + " con train_len: " + str(
                train_lens) + " as no best len are defined ... SKIPPING!")
        else:
            for train_len in train_lens:
                print("Running: " + str(stock_dir) + " con train_len: " + str(train_len))
                command = "python3.6 /home/toniotonia47/PycharmProjects/mystuff/ingestion/main.py --datapath "
                # command = "python3.5 /home/edge7/PycharmProjects/MlStock/ingestion/main.py --datapath "
                command += str(stock_dir)
                target = str(stock_dir).split("/")[-1]
                command += " --target " + target
                command += " --train_len " + str(train_len)
                command += " --predict yes"
                with open(os.devnull, 'w') as devnull:
                    subprocess.run(command, check=True, shell=True, stdout=devnull, stderr=devnull)
