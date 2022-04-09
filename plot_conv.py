import numpy as np
from .utils import get_padding
import matplotlib.pyplot as plt

def plot_conv(array, ks, stride=1, offset_x=0, offset_y=0, mode="SAME", connect=False, name="conv", gap=6):
    height, width, channels = array.shape
    pad = get_padding(ks, mode)
    
    fig = plt.figure(figsize=(7, 5))
    ax = fig.gca(projection='3d')
    ax._axis3don = False
    ax.set_zlim3d(0, gap)
    max_size = max(height + pad[1] + pad[3], width + pad[0] + pad[2])
    ax.set_ylim(0, max_size + 1)
    ax.set_xlim(0, max_size + 1)

    # input surface
    x = np.linspace(0, width, width + 1)
    y = np.linspace(0, height, height + 1)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    surf = ax.plot_surface(X, Y, Z, alpha=0.3)

    # padding surface
    x = np.linspace(-pad[0], width + pad[2], width + 1 + pad[0] + pad[2])
    y = np.linspace(-pad[1], height + pad[3], height + 1 + pad[1] + pad[3])
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    surf = ax.plot_surface(X, Y, Z, alpha=0.3, color="white", lw=0.5, edgecolors="white")
    
    # lines
    sx, sy, sz = -pad[0] + offset_x, -pad[1] + offset_y, 0
    p1 = (sx, sy, sz)
    p2 = (sx, sy + ks[1], sz)
    p3 = (sx + ks[0], sy + ks[1], sz)
    p4 = (sx + ks[0], sy, sz)
    frame_line(ax, p1, p2)
    frame_line(ax, p2, p3)
    frame_line(ax, p3, p4)
    frame_line(ax, p4, p1)
    
    x = np.linspace(sx, sx + ks[0], sx + ks[0] + 1)
    y = np.linspace(sy, sy + ks[1], sy + ks[1] + 1)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X) + sz
    surf = ax.plot_surface(X, Y, Z, alpha=0.2, color="red")
    
    # output surface
    res_x_len = int((width + pad[0] + pad[2] - ks[0]) / stride + 1)
    res_y_len = int((height + pad[1] + pad[3] - ks[1]) / stride + 1)
    center_x = (width) / 2
    center_y = (height) / 2
    
    x = np.linspace(center_x - res_x_len / 2, center_x + res_x_len / 2, res_x_len + 1)
    y = np.linspace(center_y - res_y_len / 2, center_y + res_y_len / 2, res_y_len + 1)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X) + gap
    surf = ax.plot_surface(X, Y, Z, alpha=0.3, color="green", lw=0.5, edgecolors="white")

    sx, sy, sz = center_x - res_x_len / 2 + offset_x, center_y - res_y_len / 2 + offset_y, gap
    p1_ = (sx, sy, sz)
    p2_ = (sx, sy + 1, sz)
    p3_ = (sx + 1, sy + 1, sz)
    p4_ = (sx + 1, sy, sz)
    frame_line(ax, p1_, p2_)
    frame_line(ax, p2_, p3_)
    frame_line(ax, p3_, p4_)
    frame_line(ax, p4_, p1_)

    if connect:
        frame_line(ax, p1, p1_)
        frame_line(ax, p2, p2_)
        frame_line(ax, p3, p3_)
        frame_line(ax, p4, p4_)
        
#     plt.title(name + " (%d, %d) -> (%d, %d)" % (height, width, res_y_len, res_x_len))
    # plt.savefig("./plots/conv/%s.png" % name)
    
def frame_line(ax, p1, p2, alpha=0.4):
    line_color = "blue"
    lw = 1.5
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zs=[p1[2], p2[2]], lw=lw, color=line_color, alpha=alpha, linestyle='--')
    
    
