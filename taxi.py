import numpy as np
import lmdb
import sys
from math import sqrt, cos, acos, sin, pi

R = 6371

sys.path.insert(0, "/home/lancy/caffe-master/python")


import caffe

sys.stdin = open("shanghai_taxi.csv", "r")

N = 1400
WIDTH = 512
HEIGHT = 512
maxLon, minLon = 122.236892, 120.930592
maxLat, minLat = 31.91274, 30.500367

max_dist = 0

eof = False


def make_lmdb(filename, times, calMax=False):

    global eof
    global max_dist
    X = np.zeros((times, 1, WIDTH, HEIGHT), dtype=np.double)
    y = np.zeros((times, 1, 1, 1), dtype=np.double)
    map_size = X.nbytes * 10
    input_env = lmdb.open(filename + "_input_lmdb", map_size=map_size)
    output_env = lmdb.open(filename + "_output_lmdb", map_size=map_size)

    count = 0
    for i in range(times):

        raw = raw_input()
        lastLon, lastLat, lastPsg = -1, -1, 0
        dist = 0
        while len(raw) > 10:
            msg = raw.split(",")
            psg = 1 - int(msg[5])
            lon = float(msg[3])
            lon = lon / 180 * pi
            lat = float(msg[4])
            lat = lat / 180 * pi
            
            if (lastPsg == 1) and (psg == 1):
                try:
                    dist += sqrt((lon - lastLon)**2 + (lat - lastLat)**2)
                except Exception as e:
                    raise e
            lastPsg = psg
            lastLon = lon
            lastLat = lat

            if (calMax is False):
                lon = int((lon - minLon) / (maxLon - minLon) * WIDTH)
                lat = int((lat - minLat) / (maxLat - minLat) * HEIGHT)
                if (lon >= 0 and lon < WIDTH and lat >= 0 and lat < HEIGHT):
                    X[i][0][lon][lat] = 1

            try:
                raw = raw_input()
            except Exception:
                eof = True
            if eof is True:
                break

        print i, dist

        if calMax:
            if (dist < 10):
                max_dist = max(max_dist, dist)
        else:
            y[i][0][0][0] = dist / max_dist

            if dist < 10: 

                str_id = '{:08}'.format(count)
# input
                with input_env.begin(write=True) as txn:
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = X.shape[1]
                    datum.height = X.shape[2]
                    datum.width = X.shape[3]
                    datum.data = X[i].tobytes()

                    txn.put(str_id.encode("ascii"), datum.SerializeToString())

# output
                with output_env.begin(write=True) as txn:
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = y.shape[1]
                    datum.height = y.shape[2]
                    datum.width = y.shape[3]
                    datum.data = y[i].tobytes()

                    txn.put(str_id.encode("ascii"), datum.SerializeToString())
                    
                count += 1

        if eof is True:
            break


make_lmdb("train", 1400, True)
make_lmdb("test", 300, True)
