import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import convolve2d

source = cv2.imread("res06.jpg")
target = cv2.imread("res05.jpg")
red_mask = cv2.imread("white_mask.png")
target = target[:400, 100:800, :]
red_mask = red_mask[:400, 100:800, :]
target = np.flip(target, axis=1)
red_mask = np.flip(red_mask, axis=1)
mask = red_mask[:,:,2] > 200
mask[:150, :100] = 0
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
mask = 1 - mask
alpha = 0.8
resized_target = cv2.resize(target.astype('uint8'), dsize=(0,0), fx=alpha, fy=alpha)
resized_mask =  cv2.resize(mask.astype('uint8'), dsize=(0,0), fx = alpha, fy=alpha)

H, W = resized_mask.shape
startX = 15
startY = 30
rectangle = source[startX:startX+H, startY:startY+W, :]


def solve_equations(lap, img, mask):
    lap = lap.astype(float)
    img = img.astype(float)
    mask = mask.astype(float)
    H, W = mask.shape
    num_px = H * W
    index = np.arange(num_px)
    index = np.reshape(index, (H, W))
    mat_data = []
    mat_row = []
    mat_col = []
    known = np.zeros((num_px))
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if mask[i, j] == 1:
                if mask[i - 1:i + 2, j - 1:j + 2].sum() == 9:
                    continue
                mat_row.append(index[i, j])
                mat_col.append(index[i, j])
                mat_data.append(1)
                known[index[i, j]] = img[i, j]
            else:
                mat_row.append(index[i, j])
                mat_col.append(index[i, j])
                mat_data.append(-4)

                mat_row.append(index[i, j])
                mat_col.append(index[i, j - 1])
                mat_data.append(1)

                mat_row.append(index[i, j])
                mat_col.append(index[i, j + 1])
                mat_data.append(1)

                mat_row.append(index[i, j])
                mat_col.append(index[i - 1, j])
                mat_data.append(1)

                mat_row.append(index[i, j])
                mat_col.append(index[i + 1, j])
                mat_data.append(1)

                known[index[i, j]] = lap[i, j]
    M = coo_matrix((mat_data, (mat_row, mat_col)), shape=(num_px, num_px))
    known = np.array(known)
    ans = lsqr(M, known)[0]
    ans = np.reshape(ans, (H, W))
    ans = np.where(mask, img, ans)
    return ans

lap_target = cv2.Laplacian(resized_target.astype('uint8'), ddepth=3)
r_lap = lap_target[:,:,0]
g_lap = lap_target[:,:,1]
b_lap = lap_target[:,:,2]

r_ans = solve_equations(r_lap, rectangle[:,:,0], resized_mask)
g_ans = solve_equations(g_lap, rectangle[:,:,1], resized_mask)
b_ans = solve_equations(b_lap, rectangle[:,:,2], resized_mask)

rgb_stacked = np.stack([r_ans, g_ans, b_ans], axis=2)

source_cop = source.copy()
source_cop[startX:startX+H, startY:startY+W, :] = np.minimum(np.maximum(rgb_stacked, 0), 255)

cv2.imwrite("res07.jpg", np.flip(source_cop, axis=2))