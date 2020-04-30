import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_route(gt, out, c_gt='g', c_out='r'):
    x_idx = 0
    y_idx = 2
    x = [v for v in gt[:, x_idx]]
    y = [v for v in gt[:, y_idx]]
    plt.plot(x, y, color=c_gt, label='Ground Truth')

    x = [v for v in out[:, x_idx]]
    y = [v for v in out[:, y_idx]]
    plt.plot(x, y, color=c_out, label='DeepVO')
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()


def visualize_route(img_arr, length):
    fig = plt.figure()
    ims = []
    for i in range(length):
        image = img_arr[i]
        im = plt.imshow(image, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()


def plot_both(img_arr, length, gt, out):
    fig = plt.figure(1)
    ims = []
    for i in range(length):
        image = img_arr[i]
        im = plt.imshow(image, animated=True)
        ims.append([im])

    fig2 = plt.figure(2)
    ax = plt.axes(ylim=(-2000, 2000), xlim=(-2000, 2000))
    line1, = ax.plot([], [], lw=2, color='g')
    line2, = ax.plot([], [], lw=2, color='r')

    x_idx = 0
    y_idx = 2

    x_gt = [v for v in gt[:, x_idx]]
    y_gt = [v for v in gt[:, y_idx]]

    x_out = [v for v in out[:, x_idx]]
    y_out = [v for v in out[:, y_idx]]

    ax.set_xlim(min(x_gt) - 100, max(x_gt) + 100)
    ax.set_ylim(min(y_gt) - 100, max(y_gt) + 100)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return [line1, line2]

    def animate(idx):
        x_data_gt = x_gt[:idx]
        y_data_gt = y_gt[:idx]
        x_data_out = x_out[:idx]
        y_data_out = y_out[:idx]

        line1.set_data(x_data_gt, y_data_gt)
        line2.set_data(x_data_out, y_data_out)
        return [line1, line2]

    ani1 = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
    ani2 = animation.FuncAnimation(fig2, animate, init_func=init, frames=length,
                                   interval=60, blit=True, repeat_delay=1000, repeat=True)
    # plt.plot(x, y, color=c_gt, label='Ground Truth')

    # x = [v for v in out[:, x_idx]]
    # y = [v for v in out[:, y_idx]]
    # plt.plot(x, y, color=c_out, label='DeepVO')
    # plt.gca().set_aspect('equal', adjustable='datalim')

    plt.show()
