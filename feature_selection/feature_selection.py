import argparse
import pathlib

if __name__ == "__main__":
    # Getting path where CSVs are stored
    parser = argparse.ArgumentParser("Ml Forex")
    parser.add_argument("--datapath", help="complete path to CSVs file directory")

    args = parser.parse_args()
    datapath = args.datapath
    flist = [p for p in pathlib.Path(datapath).iterdir() if p.is_file()]

    feature_selection = dict()

    for file in flist:

        with file.open() as f:
            print('Reading file: ' + str(file) + '\n')
            l = f.readline()
            while l:
                if l == ' Writing Feature IMPORTANCE:\n':
                    print('Found')
                    f.readline()
                    l = f.readline()
                    while l:
                        arr = l.split("   ")
                        feature = arr[0]
                        val = float(arr[1])
                        res = feature_selection.get(feature, [])
                        res.append(val)
                        feature_selection[feature] = res
                        l = f.readline()
                l = f.readline()

    l = []
    for name, lis in feature_selection.items():
        tot = float(sum(lis))
        count = float(len(lis))
        mean = tot / count
        l.append((name, mean))

    tw = sorted(l, key=lambda tup: tup[1], reverse=True)
    to_write = ''
    for t in tw:
        to_write += str(t[0]) + ':   ' + str(t[1])
        to_write += '\n'
    f = open(datapath + 'response', 'a')
    f.write(to_write)
    f.close()