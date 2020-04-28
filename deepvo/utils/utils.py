from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_image(path):
    return Image.open(path).convert("RGB")


def Log(tag, message):
    print(f'[{tag}] {message}')


def isRotationMatrix(R):
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - np.dot(R.T, R))
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z], dtype=np.float32)


def get6DoFPose(p):
    pos = np.array([p[3], p[7], p[11]])
    R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    angles = rotationMatrixToEulerAngles(R)
    return np.concatenate((pos, angles))


def visualize(img_arr, length):
    fig = plt.figure()
    ims = []
    for i in range(length):
        image = img_arr[i, :3, :, :]
        image = np.moveaxis(image, 0, 2)
        im = plt.imshow(image, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()
