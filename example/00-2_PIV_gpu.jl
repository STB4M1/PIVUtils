using StatsBase, CairoMakie, Images, Logging, FFTW, JSON, Plots
using CUDA
using ImageUtils
using PIVUtils

img1 = read_img_gray_float64("./00_images/rect_particles_frame1.png")
img2 = read_img_gray_float64("./00_images/rect_particles_frame2.png")

win_h = 64
win_w = 64

# === PIV 実行 ===
println("PIV begin.")
@time vec_dx, vec_dy, vec_R = PIV_gpu(img1, img2;
        interro_win_w = win_w,
        interro_win_h = win_h,
        search_factor = 2)
println("PIV end.")

# === グリッド座標 ===
height, width = size(img1)
interro_win_w = win_w
interro_win_h = win_h

step_x = interro_win_w ÷ 2
step_y = interro_win_h ÷ 2

interro_cx_list = collect(interro_win_w÷2 : step_x : width - interro_win_w÷2)
interro_cy_list = collect(interro_win_h÷2 : step_y : height - interro_win_h÷2)

X = repeat(interro_cx_list', length(interro_cy_list), 1)
Y = repeat(interro_cy_list, 1, length(interro_cx_list))

h, w = size(img1)
arrow_scale = 12.0

p = Plots.plot(aspect_ratio=:equal, legend=false)
Plots.xlims!(p, (0, w))
Plots.ylims!(p, (0, h))
# Plots.plot!(p; yflip=true)

Plots.quiver!(p, X, Y,
    quiver=(arrow_scale .* vec_dx, arrow_scale .* vec_dy),
    color=:royalblue, linewidth=1.5,
)

savefig(p, "piv_vectors_only_gpu.png")

# ====== PIV ベクトルを .dat 保存 ======

open("piv_vector_field_gpu.dat", "w") do io
    for j in 1:size(vec_dx)[1]
        for i in 1:size(vec_dx)[2]
            x  = X[j, i]
            y  = Y[j, i]
            dx = vec_dx[j, i]
            dy = vec_dy[j, i]

            println(io, "$x $y $dx $dy")
        end
    end
end
println("PIV vector field saved to piv_vector_field.dat")

# cueffs = get_distortion_coefficients_local(img1, img2, verbose = true,  gridSize = 128, intrSize = 128, srchSize = 256)
# save_coefficients(cueffs, "coefficients_cpu.dat")
