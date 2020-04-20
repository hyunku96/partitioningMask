import numpy as np
import matplotlib.pyplot as plt
import cv2


with open('a.dat') as file:
    out = file.read().splitlines()

bin = []
for i in range(len(out[0])):
    bin.append(ord(out[0][i]))

frame = 46
bin = np.reshape(bin, (115, 240//16, 352//16))
print(bin[frame])  # 46th frame CU partitioning

# binary mask
mask = np.zeros((240, 352))
for i in range(240//16):
    for j in range(352//16):
        if bin[frame, i, j] == 0 and bin[frame, i + 3, j + 3] == 0:
            if i % 4 == 0 and j % 4 == 0:
                mask[i*16:i*16+16*4, j*16:j*16+16*4] = 255
                mask[i*16+1:i*16+16*4-1, j*16+1:j*16+16*4-1] = 0
                j = j + 4
        elif bin[frame, i, j] == 1 and bin[frame, i + 1, j + 1] == 1:
            if i % 2 == 0 and j % 2 == 0:
                mask[i*16:i*16 + 16 * 2, j*16:j*16 + 16 * 2] = 255
                mask[i*16 + 1:i*16 + 16 * 2 - 1, j*16 + 1:j*16 + 16 * 2 - 1] = 0
                j = j + 2
        elif bin[frame, i, j] == 2:
            mask[i*16:i*16 + 16, j*16:j*16 + 16] = 255
            mask[i*16 + 1:i*16 + 16 - 1, j*16 + 1:j*16 + 16 - 1] = 0
        elif bin[frame, i, j] == 3:
            mask[i*16:i*16 + 16, j*16:j*16 + 16] = 255
            mask[i*16 + 1:i*16 + 16 - 1, j*16 + 1:j*16 + 16 - 1] = 0
            mask[i*16:i*16 + 16, j*16+8] = 255
            mask[i*16+8, j*16:j*16 + 16] = 255

cv2.imwrite("{0}th frame CU partition binary mask.png".format(frame), mask)


# mean mask
width = 352
height = 240

stream = open('rec.yuv', 'rb')

# Load the Y (luminance) data from the stream
Y = np.fromfile(stream, dtype=np.uint8, count=width*height*(frame+1)+(width//2*height//2*2)*frame)
Y = Y[frame*(width*height+width//2*height//2*2):frame*(width*height+width//2*height//2*2)+width*height].reshape(height, width)
# Load the UV (chrominance) data from the stream, and double its size
U = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).\
        reshape((height//2, width//2)).\
        repeat(2, axis=0).repeat(2, axis=1)
V = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).\
        reshape((height//2, width//2)).\
        repeat(2, axis=0).repeat(2, axis=1)
# Stack the YUV channels together, crop the actual resolution, convert to
# floating point for later calculations, and apply the standard biases
YUV = np.dstack((Y, U, V))[:height, :width, :].astype(np.float)
YUV[:, :, 0]  = YUV[:, :, 0]  - 16   # Offset Y by 16
YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
# YUV conversion matrix from ITU-R BT.601 version (SDTV)
# Note the swapped R and B planes!
#              Y       U       V
M = np.array([[1.164,  2.017,  0.000],    # B
              [1.164, -0.392, -0.813],    # G
              [1.164,  0.000,  1.596]])   # R
# Take the dot product with the matrix to produce BGR output, clamp the
# results to byte range and convert to bytes
BGR = YUV.dot(M.T).clip(0, 255).astype(np.uint8)

mask = np.zeros((240, 352))
for i in range(240//16):
    for j in range(352//16):
        if bin[frame, i, j] == 0 and bin[frame, i + 3, j + 3] == 0:
            if i % 4 == 0 and j % 4 == 0:
                mask[i*16:i*16+16*4, j*16:j*16+16*4] = np.mean(BGR[i*16:i*16+16*4, j*16:j*16+16*4])
                j = j + 4
        elif bin[frame, i, j] == 1 and bin[frame, i + 1, j + 1] == 1:
            if i % 2 == 0 and j % 2 == 0:
                mask[i*16:i*16 + 16 * 2, j*16:j*16 + 16 * 2] = np.mean(BGR[i*16:i*16+16*2, j*16:j*16+16*2])
                j = j + 2
        elif bin[frame, i, j] == 2:
            mask[i*16:i*16 + 16, j*16:j*16 + 16] = np.mean(BGR[i*16:i*16+16, j*16:j*16+16])
        elif bin[frame, i, j] == 3:
            mask[i*16:i*16+8, j*16:j*16+8] = np.mean(BGR[i*16:i*16+8, j*16:j*16+8])
            mask[i*16+8:i*16+16, j*16:j*16+8] = np.mean(BGR[i*16+8:i*16+16, j*16:j*16+8])
            mask[i*16:i*16+8, j*16+8:j*16+16] = np.mean(BGR[i*16:i*16+8, j*16+8:j*16+16])
            mask[i*16+8:i*16+16, j*16+8:j*16+16] = np.mean(BGR[i*16+8:i*16+16, j*16+8:j*16+16])

cv2.imwrite("{0}th frame CU partition mean mask.png".format(frame), mask)