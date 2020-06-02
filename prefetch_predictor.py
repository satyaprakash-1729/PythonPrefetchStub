import numpy as np
import keras
import json
from keras_self_attention import SeqSelfAttention


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

def prefetch_predict(lastNData) -> int:
  N = 20
  if len(lastNData) < 2*N+3:
    return -1

  lastNPCs = lastNData[:N]
  lastNAddrs = lastNData[N:2*N]
  topk = lastNData[2*N]
  idx_to_delta = lastNData[2*N+1]
  model = lastNData[2*N+2]

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
  topkidxs = y_pred.argsort()[:, -topk:][0]

  return [idx_to_delta[str(idx)] for idx in topkidxs]
  
# idx_to_delta = initialize_idx_to_delta()
# model1 = initialize_model()
# print(prefetch_predict([6167872, 6628928, 6605932, 6145648, 6524736, 6605932, 6525010, 6525111,
#  6525202, 6605932, 6159894, 6192705, 6605932, 6173729, 6174606, 4834393, 6605932,
# 5547136, 6258432, 6605932, 11137690676544, 11137690678848, 11137690748160, 279264050869824,
#  279264050859840, 11137690748224, 11137690771520, 11137690771584, 11137690771712, 11137690748288,
#   11137690709504, 11137690705472, 11137690748352, 11137690776576, 11137690777472, 11137690780736,
#    11137690748416, 11137690711168, 11137690717952, 11137690748480, 10, idx_to_delta, model1]))