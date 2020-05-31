from typing import List
from sklearn.cluster import KMeans
import json
import tensorflow as tf
import numpy as np


def prefetch_predict(lastNPCs: List[int], lastNAddrs: List[int], topk: int, ROOT_DIR: str = "./") -> int:
  def get_cluster_id(center1, center2, point):
    if abs(center1-point) < abs(center2-point):
      return 0
    else:
      return 1

  idx_to_delta = None
  with open(ROOT_DIR + 'delta_idx_map.json') as json_file:
    idx_to_delta = json.load(json_file)

  N = 5
  if len(lastNPCs) != N or len(lastNAddrs) != N:
    return -1

  f = open(ROOT_DIR + "model_for_cpp.json")
  model = tf.keras.models.model_from_json(f.read())
  f.close()

  ip1 = 94602865943828.0
  ip2 = 140119319091961.0

  addr11 = 140118976570040.0
  addr12 = 140732090279160.0

  addr21 = 94602865942592.0
  addr22 = 94602868043752.0

  model.load_weights(ROOT_DIR + "model_for_cpp_weights.h5")
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  
  center1 = 1.40618487e+14
  center2 = 9.46028667e+13
  ips = np.array(lastNPCs)
  ips = [(ip-ip1)/(ip2-ip1) for ip in ips]

  addrs = np.array(lastNAddrs)
  cluster_ids = [get_cluster_id(center1, center2, addr) for addr in addrs]

  for i in range(len(addrs)):
    if cluster_ids[i] == 0:
      addrs[i] = (addrs[i]-addr11) / (addr12-addr11)
    else:
      addrs[i] = (addrs[i]-addr21) / (addr22-addr21)

  X = np.array([list(xx) for xx in zip(ips, cluster_ids, addrs)])
  X = X.reshape(1, N, 3)
  y_pred = model.predict(X)

  topkidxs = y_pred.argsort()[:, -topk:][0]

  return [idx_to_delta[str(idx)] for idx in topkidxs]
  

print(prefetch_predict([9.46887013e-12, 8.85380658e-12, 9.02966590e-12, 9.18332077e-12, 9.31521527e-12],
  [0.99999919, 0.99999919, 0.99999926, 0.99999919, 0.99999919], 10))