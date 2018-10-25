import numpy as np
import pickle, cv2
from settings import *


bbox = np.array(((1047, 405, 1259, 510))) # 0.79
#bbox = np.array(((822, 413, 941, 488)))   # 0.80
#[[ 412.58627  822.1018   488.40225  941.1221 ]
# [ 404.99207 1047.3772   510.309   1258.7302 ]]
print(bbox[0]/2+bbox[2]/2)

# before reshape
temp = np.array((bbox[0]/2+bbox[2]/2, bbox[3]))
print(temp)
print(temp.shape)

# after
pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
print('pos',pos)
print(pos.shape)


with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
		perspective_data = pickle.load(f)

perspective_transform = perspective_data["perspective_transform"]
pixels_per_meter = perspective_data['pixels_per_meter']
orig_points = perspective_data["orig_points"]

print(perspective_transform)
print(perspective_transform.shape)

dst0 = cv2.perspectiveTransform(pos, perspective_transform)
print('dst0', dst0)
print(dst0.shape)

dst = cv2.perspectiveTransform(pos, perspective_transform).reshape(-1, 1)
print('dst', dst)
print(dst.shape)

warped_size = UNWARPED_SIZE
pix_per_meter = pixels_per_meter
print('warped_size[1]', warped_size[1])
print('pix_per_meter[1]', pix_per_meter[1])

d = np.array((warped_size[1]-dst[1])/pix_per_meter[1])
print('d', d)
print(d.shape)

d1d = np.atleast_1d(d)
print(d[0])