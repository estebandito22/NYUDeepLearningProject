import os
import json
import argparse
import numpy as np

def sample(datajson, size=100):
    size = int(size)
    cur_dir = os.getcwd()
    inputfolder = "input"
    datafolder = "MSRVTT"
    jsonpath = os.path.join(cur_dir, inputfolder, datafolder, datajson)

    data = json.load(open(jsonpath, "r"))
    assert(len(data["data"]) >= size)

    np.random.shuffle(data["data"])
    data_sample = json.dumps({"data": data["data"][:size]})

    sample = open(jsonpath+".sample", "w")
    sample.write(data_sample)

if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datajson", required=True,
    	help="name of data.json file")
    ap.add_argument("-n", "--size", default=100, required=False,
    	help="sample size")
    args = vars(ap.parse_args())
    print(args['datajson'])

    sample(args['datajson'], args['size'])
