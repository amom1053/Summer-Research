import numpy as np
import matplotlib.pyplot as plt

def plot_1d_data(data):
    if data is None:
        print("plot_1d_data: Input data is None")
        return
    x = np.arange(len(data))
    plt.plot(x, data)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Array Data Plot')
    plt.grid(True)
    plt.show()

def plot_two_1d_data(data_org, data_proc):
    if (data_org is None) or (data_proc is None):
        print("plot_two_1d_data: Input data is None")
        return
    x = np.arange(len(data_org))
    plt.plot(x, data_org, color='r', label='original')
    plt.plot(x, data_proc, color='b', label='processed')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Array Data Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_image(img):
    if img is None:
        print("show_image: Input data is None")
        return
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

def show_two_images(img_org, img_proc):
    if (img_org is None) or (img_proc is None):
        print("show_two_images: Input data is None")
        return
    fig = plt.figure()
    fig.add_subplot(1,2, 1)
    plt.imshow(img_org, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_proc, cmap='gray', vmin=0, vmax=255)
    plt.show()