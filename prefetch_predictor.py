import numpy as np
import keras
import json
from keras_self_attention import SeqSelfAttention
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def initialize_idx_to_delta(ROOT_DIR="./"):
  idx_to_delta = None
  with open(ROOT_DIR + 'delta_idx_map.json') as json_file:
    idx_to_delta = json.load(json_file)
  return idx_to_delta

def initialize_model(ROOT_DIR="./"):
  f = open(ROOT_DIR + "model_for_cpp_self_attn.json")
  model = keras.models.model_from_json(f.read(), custom_objects={'SeqSelfAttention': SeqSelfAttention})
  f.close()
  model.load_weights(ROOT_DIR + "model_for_cpp_self_attn_weights.h5")
  model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  return model

def prefetch_predict(lastNData):
  N = 10
  if len(lastNData) < 2*N+4:
    return []

  lastNPCs = lastNData[:N]
  lastNAddrs = lastNData[N:2*N]
  topk = lastNData[2*N]
  idx_to_delta = lastNData[2*N+1]
  model = lastNData[2*N+2]
  LOG2_PAGE_SIZE = lastNData[2*N+3]

  ip1 = 4195632.0
  ip2 = 8208480.0

  addr11 = 96935917361984.0
  addr12 = 195743444536640.0

  addr21 = 5610901135424.0
  addr22 = 80904477544064.0

  addr31 = 222418542097984.0
  addr32 = 279264050879424.0

  center1 = 1.61729982e+14
  center2 = 3.12041036e+13
  center3 = 2.53453964e+14

  ips = np.array(lastNPCs)
  ips = [(ip-ip1)/(ip2-ip1) for ip in ips]

  def get_cluster_id(center1, center2, center3, point):
    arr = np.array([abs(center1-point), abs(center2-point), abs(center3-point)])
    return np.argmin(arr)

  addrs = np.array(lastNAddrs)
  cluster_ids = [get_cluster_id(center1, center2, center3, addr) for addr in addrs]

  addrs[cluster_ids==0] = (addrs[cluster_ids==0] - addr11)/(addr12-addr11)
  addrs[cluster_ids==1] = (addrs[cluster_ids==1] - addr21)/(addr22-addr21)
  addrs[cluster_ids==2] = (addrs[cluster_ids==2] - addr31)/(addr32-addr31)

  X = np.array([list(xx) for xx in zip(ips, cluster_ids, addrs)])
  X = X.reshape(1, N, 3)

  y_pred = model.predict(X)
  idxs = y_pred.argsort()[0]
  topkidxs = []
  lastaddr = int(lastNData[2*N-1])
  for i in range(len(idxs)):
    if len(topkidxs) == topk:
      break
    delta1 = int(idx_to_delta[str(idxs[i])])
    if ((lastaddr + delta1) >> LOG2_PAGE_SIZE) == (lastaddr >> LOG2_PAGE_SIZE):
      topkidxs.append(delta1)

  return topkidxs
  
idx_to_delta = initialize_idx_to_delta()
model1 = initialize_model()

args = []
args += list(map(int, sys.argv[1].split(",")))
args += list(map(int, sys.argv[2].split(",")))
args += [int(sys.argv[3])]

args += [idx_to_delta]
args += [model1]
args += [int(sys.argv[4])]

print(prefetch_predict(args))

