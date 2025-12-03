using Random
using FSUtils          # make_dirs
using ImageUtils       # save_gray_standard
using Printf

# ============================================
# Box–Muller 法（正規分布生成）
# ============================================
function rand_normal(μ, σ)
    r1 = rand()
    r2 = rand()
    z = sqrt(-2 * log(r1)) * cos(2π * r2)
    return μ + σ*z
end

# ============================================
# 粒子直径生成（正規分布 → 任意 bin で丸め）
# ============================================
function generate_diameters(N; μ=50.0, σ=15.0, bin=1.0)
    d = [rand_normal(μ, σ) for _ in 1:N]
    d_clamped = clamp.(d, 5.0, 120.0)
    d_round = round.(d_clamped ./ bin) .* bin
    return d_round
end

# ============================================
# CPU 描画：粒子をガウシアンで描画（GPU と同じ処理）
# ============================================
function draw_particles_cpu!(img, centers, W, H)
    @inbounds for (x0, y0, d, a) in centers
        r = Int(round(d/2))
        σ = d / 4.0

        for j in -r:r
            yy = y0 + j
            if !(1 <= yy <= H); continue; end

            for i in -r:r
                xx = x0 + i
                if !(1 <= xx <= W); continue; end

                rr2 = i*i + j*j
                if rr2 <= (r+1)^2
                    val = a * exp(-rr2 / (2 * σ * σ))
                    img[yy, xx] += val
                end
            end
        end
    end
end

# ============================================
# 粒子配置（重なり禁止）
# ============================================
function place_particles(W, H, diameters)
    centers = Vector{Tuple{Int,Int,Float32,Float32}}()

    println("  - 粒子配置開始...")
    for (idx, d) in enumerate(diameters)
        r = Int(round(d/2))
        placed = false
        tries = 0
        max_try = 6000

        while !placed && tries < max_try
            tries += 1
            x = rand(1:W)
            y = rand(1:H)

            ok = true
            for (cx, cy, cd, _) in centers
                if hypot(cx - x, cy - y) < (cd/2 + r)
                    ok = false
                    break
                end
            end

            if ok
                a = Float32(rand(128:255))
                push!(centers, (x, y, Float32(d), a))
                placed = true
            end
        end

        if !placed
            @warn "粒子 $idx を配置できずスキップ"
        end
    end

    println("  - 粒子配置完了：$(length(centers)) / $(length(diameters))")
    return centers
end

# ============================================
# Taylor–Green vortex（長方形でも綺麗な渦）
# ============================================
# function generate_velocity_field(W, H)
#     u = zeros(Float32, H, W)
#     v = zeros(Float32, H, W)

#     for j in 1:H, i in 1:W
#         x = (i - 0.5) / W * 2π
#         y = (j - 0.5) / H * 2π

#         u[j, i] =  sin(x) * cos(y)
#         v[j, i] = -cos(x) * sin(y)
#     end

#     return u, v
# end

function generate_velocity_field(W, H)
    u = zeros(Float32, H, W)
    v = zeros(Float32, H, W)

    for j in 1:H, i in 1:W
        # C 側と同じ：i, j を 0 ベースにしてから 2π の周期へ
        x = 2π * (i - 1) / W
        y = 2π * (j - 1) / H

        # velosity_vectors.cpp と同じ定義
        u[j, i] =  cos(x) * sin(y)
        v[j, i] = -sin(x) * cos(y)
    end

    return u, v
end

# ============================================
# 粒子移動（u,v のインデックスを [y,x] に統一）
# ============================================
function move_particles(centers, u, v; t=10.0)
    moved = Vector{Tuple{Int,Int,Float32,Float32}}()
    Hy, Wx = size(u)

    for (x, y, d, a) in centers
        xx = clamp(x, 1, Wx)
        yy = clamp(y, 1, Hy)

        dx = round(Int, u[yy, xx] * t)
        dy = round(Int, v[yy, xx] * t)

        push!(moved, (x + dx, y + dy, d, a))
    end
    return moved
end

# ============================================
# displacement.dat 出力
# ============================================
function save_displacement(path, centers, centers2)
    open(path, "w") do io
        println(io, "# id x0 y0 x1 y1 dx dy diameter amplitude")

        for i in eachindex(centers)
            x0, y0, d, a = centers[i]
            x1, y1, _, _ = centers2[i]

            dx = x1 - x0
            dy = y1 - y0

            @printf(io, "%d %d %d %d %d %d %d %.3f %.3f\n",
                i, x0, y0, x1, y1, dx, dy, d, a)
        end
    end
end

# ============================================
# 共通処理（任意サイズで計算）
# ============================================
function run_simulation(W, H, PART_NUM, prefix)
    out_dir = "./00_images"
    make_dirs(out_dir)

    println("1. 粒子径生成...")
    diameters = generate_diameters(PART_NUM; μ=20.0, σ=5.0, bin=1.0)

    println("2. CPUで frame1 描画...")
    centers = place_particles(W, H, diameters)
    img1 = zeros(Float32, H, W)
    draw_particles_cpu!(img1, centers, W, H)
    img1 ./= maximum(img1)
    save_gray_standard(img1, joinpath(out_dir, "$(prefix)_frame1.png"))

    println("3. ベクトル場生成 (Taylor–Green)...")
    u, v = generate_velocity_field(W, H)

    println("4. 粒子移動計算...")
    centers2 = move_particles(centers, u, v; t=10.0)

    println("5. CPUで frame2 描画...")
    img2 = zeros(Float32, H, W)
    draw_particles_cpu!(img2, centers2, W, H)
    img2 ./= maximum(img2)
    save_gray_standard(img2, joinpath(out_dir, "$(prefix)_frame2.png"))

    println("6. 粒子移動データ出力...")
    save_displacement(joinpath(out_dir, "$(prefix)_displacement.dat"),
                      centers, centers2)

    println("🎉 完了！（$W × $H）")
end

# ============================================
# MAIN（長方形 + 正方形の両方を生成）
# ============================================
function main()
    # 長方形
    run_simulation(1024, 512, 500, "rect_particles")

    # 正方形（512×512）
    run_simulation(512, 512, 300, "square_particles")
end

main()
