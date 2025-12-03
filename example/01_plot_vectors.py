import numpy as np
from viscore.plotting_vector_2d import create_vector_field_2d

# dat 読み込み
data = np.loadtxt("piv_vector_field_gpu.dat")

x  = data[:, 0]
y  = data[:, 1]
dx = data[:, 2]
dy = data[:, 3]

create_vector_field_2d(
    x, y, dx, dy,
    output_image_path="02_results/piv_vector_field_gpu.png",

    arrow_scale=6.0,
    colormap="turbo",    
    arrow_linewidth=1.5,   # ← 太い外枠にする
    arrow_width=0.005,  
    x_pixel_pitch=1.0,  # μm/pixel など入れる
    y_pixel_pitch=1.0,
)


# dat 読み込み
data = np.loadtxt("piv_vector_field_cpu.dat")

x  = data[:, 0]
y  = data[:, 1]
dx = data[:, 2]
dy = data[:, 3]

create_vector_field_2d(
    x, y, dx, dy,
    output_image_path="02_results/piv_vector_field_cpu.png",

    arrow_scale=12.0,
    arrow_linewidth=1.0, 
    arrow_width=0.008, 
    colormap="turbo",
    x_pixel_pitch=1.0,  # μm/pixel など入れる
    y_pixel_pitch=1.0,
)
