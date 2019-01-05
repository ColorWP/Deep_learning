'''具有递归挤压和激励网络的单图像超分辨率'''

import h5py
import matplotlib.pyplot as plt


file_path='./lap_pry_x4_small.h5'
hf = h5py.File(file_path)

data = hf.get("data")[2].squeeze()
label_x2 = hf.get("label_x2")[2].squeeze()
label_x4 = hf.get("label_x4")[2].squeeze()
print(data.shape)
print(label_x2.shape)
print(label_x4.shape)
ax = plt.subplot("131")
ax.imshow(data)
ax = plt.subplot("132")
ax.imshow(label_x2)
ax = plt.subplot("133")
ax.imshow(label_x4)


plt.show()


