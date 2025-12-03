module PIVUtils

export remove_error_vec,
       parabolic_subpixel,
       PIV_cpu,
       cu_ncc!,
       PIV_gpu

using CUDA

#=========================== CPU ===========================#
function remove_error_vec(dx::Array{Float64,2}, dy::Array{Float64,2};
                          C1=0.0, C2=3.0)

    ys, xs = size(dx)    # ys = height direction, xs = width direction

    # 出力配列をコピーで作る（C の dx1_cal と dx1 は同じ動き）
    dx_cal = copy(dx)
    dy_cal = copy(dy)

    # ノルム d1[y][x] = sqrt(dx^2 + dy^2)
    d1 = sqrt.(dx .^ 2 .+ dy .^ 2)

    # 出力（除去後のベクトル）
    dx_out = copy(dx)
    dy_out = copy(dy)

    # 統計用配列
    U_mx = zeros(Float64, ys, xs)
    U_my = zeros(Float64, ys, xs)
    U_m  = zeros(Float64, ys, xs)

    sd_x = zeros(Float64, ys, xs)
    sd_y = zeros(Float64, ys, xs)
    sd   = zeros(Float64, ys, xs)

    ep_x = zeros(Float64, ys, xs)
    ep_y = zeros(Float64, ys, xs)
    ep   = zeros(Float64, ys, xs)

    error_count = 0

    # ==== Cコード完全移植 ====
    for j in 1:ys-1
        for i in 1:xs-1

            # --------------------------------------------
            # ① 周囲3×3の平均 U_mx, U_my, U_m
            # --------------------------------------------
            S = 0
            dx_sum = 0.0
            dy_sum = 0.0
            d_sum  = 0.0

            for y in j-1:j+1
                if y < 1 || y > ys
                    continue
                end
                for x in i-1:i+1
                    if x < 1 || x > xs
                        continue
                    end
                    dx_sum += dx_cal[y, x]
                    dy_sum += dy_cal[y, x]
                    d_sum  += d1[y, x]
                    S += 1
                end
            end

            # 中心ベクトルを除く
            dx_sum -= dx_cal[j, i]
            dy_sum -= dy_cal[j, i]
            d_sum  -= d1[j, i]
            S -= 1

            U_mx[j, i] = dx_sum / S
            U_my[j, i] = dy_sum / S
            U_m[j, i]  = d_sum  / S

            # --------------------------------------------
            # ② 標準偏差 sd_x, sd_y, sd
            # --------------------------------------------
            S2 = 0
            dev2_x_sum = 0.0
            dev2_y_sum = 0.0
            dev2_sum   = 0.0

            for y in j-1:j+1
                if y < 1 || y > ys
                    continue
                end
                for x in i-1:i+1
                    if x < 1 || x > xs
                        continue
                    end
                    dev2_x_sum += (dx_cal[y, x] - U_mx[j, i])^2
                    dev2_y_sum += (dy_cal[y, x] - U_my[j, i])^2
                    dev2_sum   += (d1[y, x]   - U_m[j, i])^2
                    S2 += 1
                end
            end

            # 中心を除く
            dev2_x_sum -= (dx_cal[j, i] - U_mx[j, i])^2
            dev2_y_sum -= (dy_cal[j, i] - U_my[j, i])^2
            dev2_sum   -= (d1[j, i]     - U_m[j, i])^2
            S2 -= 1

            sd_x[j, i] = sqrt(dev2_x_sum / S2)
            sd_y[j, i] = sqrt(dev2_y_sum / S2)
            sd[j, i]   = sqrt(dev2_sum   / S2)

            # --------------------------------------------
            # ③ 平均絶対偏差 ep_x, ep_y
            # --------------------------------------------
            S3 = 0
            d_vec_x_sum = 0.0
            d_vec_y_sum = 0.0
            d_vec_sum   = 0.0

            for y in j-1:j+1
                if y < 1 || y > ys
                    continue
                end
                for x in i-1:i+1
                    if x < 1 || x > xs
                        continue
                    end

                    d_vec_x = abs(dx_cal[y, x] - dx_cal[j, i])
                    d_vec_y = abs(dy_cal[y, x] - dy_cal[j, i])
                    d_vec   = abs(d1[y, x]     - d1[j, i])

                    d_vec_x_sum += d_vec_x
                    d_vec_y_sum += d_vec_y
                    d_vec_sum   += d_vec
                    S3 += 1
                end
            end

            # 中心を除く
            S3 -= 1

            ep_x[j, i] = sqrt(d_vec_x_sum / S3)
            ep_y[j, i] = sqrt(d_vec_y_sum / S3)
            ep[j, i]   = sqrt(d_vec_sum   / S3)

            # --------------------------------------------
            # ④ 閾値判定：外れ値を除去
            # et_x = C1 + C2*sd_x[j][i]
            # --------------------------------------------
            et_x = C1 + C2 * sd_x[j, i]
            et_y = C1 + C2 * sd_y[j, i]

            if ep_x[j, i] > et_x || ep_y[j, i] > et_y
                dx_out[j, i] = 0.0
                dy_out[j, i] = 0.0
                error_count += 1
            end
        end
    end

    println("Number of removed vectors = $error_count")

    return dx_out, dy_out
end

function parabolic_subpixel(R::Array{Float64,2}, x0::Int, y0::Int)

    h, w = size(R)

    # 境界にいたらサブピクセルはできない
    if x0 <= 1 || x0 >= w || y0 <= 1 || y0 >= h
        return (0.0, 0.0)
    end

    # 周囲の値を取得
    val_y1_x0   = R[y0 + 1, x0]
    val_y0_x0   = R[y0,     x0]
    val_y_1_x0  = R[y0 - 1, x0]

    val_y0_x1   = R[y0, x0 + 1]
    val_y0_x_1  = R[y0, x0 - 1]

    # 二次曲線の分母
    denom_y = val_y1_x0 - 2.0 * val_y0_x0 + val_y_1_x0
    denom_x = val_y0_x1 - 2.0 * val_y0_x0 + val_y0_x_1

    # ゼロ除算回避
    if denom_y == 0.0
        denom_y = 1e-12
    end
    if denom_x == 0.0
        denom_x = 1e-12
    end

    # parabolic interpolation
    delta_y = (val_y1_x0 - val_y_1_x0) / (2.0 * denom_y)
    delta_x = (val_y0_x1 - val_y0_x_1) / (2.0 * denom_x)

    return (delta_x, delta_y)
end

function PIV_cpu(before_img::Array{Float64,2}, after_img::Array{Float64,2};
             interro_win_w::Int = 32,    # interrogation window width
             interro_win_h::Int = 32,    # interrogation window heignt
             search_factor::Int = 2)     # search window = 2 × interrogation window

    height, width = size(before_img)

    # search window サイズ
    search_w = search_factor * interro_win_w
    search_h = search_factor * interro_win_h

    # 出力配列（ベクトル）
    # grid のステップは 50% overlap（Cコード仕様）
    step_x = div(interro_win_w, 2)
    step_y = div(interro_win_h, 2)

    interro_cx_num = Int((width  - interro_win_w) / step_x) + 1
    interro_cy_num = Int((height - interro_win_h) / step_y) + 1

    vec_dx = zeros(Float64, interro_cy_num, interro_cx_num)
    vec_dy = zeros(Float64, interro_cy_num, interro_cx_num)
    vec_R  = zeros(Float64, interro_cy_num, interro_cx_num)

    for gy in 1:interro_cy_num
        # y方向の interrogation window の中心位置
        interro_cy = interro_win_h ÷ 2 + (gy - 1) * step_y

        for gx in 1:interro_cx_num
            # x方向の interrogation window の中心位置
            interro_cx = interro_win_w ÷ 2 + (gx - 1) * step_x

            # ==== 1. Interrogation Window (before_img) ====
            interro = zeros(Float64, interro_win_h, interro_win_w)
            interro_sum = 0.0

            for j in 1:interro_win_h
                for i in 1:interro_win_w
                    y = (interro_cy - interro_win_h÷2) + j
                    x = (interro_cx - interro_win_w÷2) + i
                    interro[j,i] = before_img[y, x]
                    interro_sum += interro[j,i]
                end
            end

            interro_mean = interro_sum / (interro_win_w * interro_win_h)

            # variance
            interro_var = 0.0
            for j in 1:interro_win_h, i in 1:interro_win_w
                interro_var += (interro[j,i] - interro_mean)^2
            end
            interro_sd = sqrt(interro_var / (interro_win_w * interro_win_h))

            # # mean / std（Cと同じ母標準偏差）
            # interro_mean = mean(interro)
            # interro_sd   = std(interro; corrected=false)

            # ==== 2. Search Window 走査 ====
            # Cでは (Y1 - s_M/2 + M/2) 〜 (Y1 + s_M/2 - M/2)
            ymin = interro_cy - search_h÷2 + interro_win_h÷2
            ymax = interro_cy + search_h÷2 - interro_win_h÷2
            xmin = interro_cx - search_w÷2 + interro_win_w÷2
            xmax = interro_cx + search_w÷2 - interro_win_w÷2

            # NCC マップ（Cの R_fg[J][I]）
            R = fill(0.0, search_h, search_w)

            # 対応する I,J 開始位置
            I0 = xmin
            J0 = ymin

            for cand_cy = ymin:ymax
                for cand_cx = xmin:xmax

                    # candidate window
                    cand_sum = 0.0
                    cand = zeros(Float64, interro_win_h, interro_win_w)

                    for j in 1:interro_win_h
                        for i in 1:interro_win_w
                            y = (cand_cy - interro_win_h÷2) + j
                            x = (cand_cx - interro_win_w÷2) + i
                            if 1 <= y <= height && 1 <= x <= width
                                cand[j,i] = after_img[y,x]
                            else
                                cand[j,i] = 0.0
                            end
                            cand_sum += cand[j,i]
                        end
                    end

                    cand_mean = cand_sum / (interro_win_w * interro_win_h)

                    # cand_sd
                    cand_var = 0.0
                    for j in 1:interro_win_h, i in 1:interro_win_w
                        cand_var += (cand[j,i] - cand_mean)^2
                    end
                    cand_sd = sqrt(cand_var / (interro_win_w * interro_win_h))


                    # 相互相関 (Cの ifcg → Cov_fg)
                    if interro_sd == 0.0 || cand_sd == 0.0
                        R[cand_cy - J0 + 1, cand_cx - I0 + 1] = 0.0
                    else
                        cov = 0.0
                        for j in 1:interro_win_h, i in 1:interro_win_w
                            cov += (interro[j,i] - interro_mean) * (cand[j,i] - cand_mean)
                        end
                        cov /= (interro_win_w * interro_win_h)

                        R[cand_cy - J0 + 1, cand_cx - I0 + 1] = cov / (interro_sd * cand_sd)
                    end
                end
            end

            # ==== 3. 最大ピーク検出（max1のみ実装）====
            maxR, linear_idx = findmax(R)
            peakY, peakX = Tuple(CartesianIndices(R)[linear_idx])

            # ==== サブピクセル補間 ====
            δx, δy = parabolic_subpixel(R, peakX, peakY)

            # ==== 画像座標への変換（Cの max1_X2, max1_Y2 のサブピクセル版）====
            cand_cx_peak = (peakX + δx - 1) + I0
            cand_cy_peak = (peakY + δy - 1) + J0

            dx = cand_cx_peak - interro_cx
            dy = cand_cy_peak - interro_cy

            vec_dx[gy, gx] = dx
            vec_dy[gy, gx] = dy
            vec_R[gy, gx]  = maxR
        end
    end
    dx_filtered, dy_filtered = remove_error_vec(vec_dx, vec_dy)

    return dx_filtered, dy_filtered, vec_R
end

#============================ GPU ==================================#


function _cu_ncc!(
    R::CuDeviceArray{Float64,2},         # 出力：NCC マップ (search_h × search_w)
    interro::CuDeviceArray{Float64,2},   # 入力：interrogation window（before_img 切り出し）
    after_img::CuDeviceArray{Float64,2}, # 入力：after_img 全体
    interro_mean::Float64,
    interro_sd::Float64,
    interro_win_w::Int,
    interro_win_h::Int,
    ymin::Int, xmin::Int,                # search window 左上の中心座標
    search_w::Int, search_h::Int,        # search window のサイズ（中心の数）
    height::Int, width::Int              # 画像全体サイズ
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # x index in search window
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # y index in search window

    if i > search_w || j > search_h
        return
    end

    # この thread が担当する候補ベクトルの中心 (cand_cx, cand_cy)
    cand_cx = xmin + (i - 1)
    cand_cy = ymin + (j - 1)

    # 窓の画素数
    N = interro_win_w * interro_win_h

    cand_sum   = 0.0
    cand_sqsum = 0.0
    bc_sum     = 0.0

    @inbounds for jj in 1:interro_win_h
        y = (cand_cy - interro_win_h ÷ 2) + jj
        for ii in 1:interro_win_w
            x = (cand_cx - interro_win_w ÷ 2) + ii

            c = 0.0
            if 1 <= y <= height && 1 <= x <= width
                c = after_img[y, x]
            end

            b = interro[jj, ii]

            cand_sum   += c
            cand_sqsum += c * c
            bc_sum     += b * c
        end
    end

    cand_mean = cand_sum / N
    cand_var  = cand_sqsum / N - cand_mean * cand_mean
    cand_sd   = cand_var > 0.0 ? sqrt(cand_var) : 0.0

    if interro_sd == 0.0 || cand_sd == 0.0
        R[j, i] = 0.0
    else
        # cov = E[bc] - E[b]E[c]
        cov = bc_sum / N - interro_mean * cand_mean
        R[j, i] = cov / (interro_sd * cand_sd)
    end

    return
end

function cu_ncc!(
    R_d::CuArray{Float64,2},
    interro_d::CuArray{Float64,2},
    after_img_d::CuArray{Float64,2},
    interro_mean::Float64,
    interro_sd::Float64,
    interro_win_w::Int,
    interro_win_h::Int,
    ymin::Int, xmin::Int,
    search_w::Int, search_h::Int,
    height::Int, width::Int
)
    threads = (16, 16)
    blocks  = (cld(search_w, threads[1]), cld(search_h, threads[2]))

    @cuda threads=threads blocks=blocks _cu_ncc!(
        R_d, interro_d, after_img_d,
        interro_mean, interro_sd,
        interro_win_w, interro_win_h,
        ymin, xmin,
        search_w, search_h,
        height, width,
    )
    return nothing
end

function PIV_gpu(before_img::Array{Float64,2}, after_img::Array{Float64,2};
                 interro_win_w::Int = 32,    # interrogation window width
                 interro_win_h::Int = 32,    # interrogation window height
                 search_factor::Int = 2)     # search window = 2 × interrogation window

    height, width = size(before_img)

    # search window サイズ
    search_w = search_factor * interro_win_w
    search_h = search_factor * interro_win_h

    # 出力ベクトル
    step_x = div(interro_win_w, 2)   # 50% overlap
    step_y = div(interro_win_h, 2)

    # interrogation window によって画像内にできるグリッドの縦横数
    interro_cx_num = Int((width  - interro_win_w) / step_x) + 1
    interro_cy_num = Int((height - interro_win_h) / step_y) + 1

    vec_dx = zeros(Float64, interro_cy_num, interro_cx_num)
    vec_dy = zeros(Float64, interro_cy_num, interro_cx_num)
    vec_R  = zeros(Float64, interro_cy_num, interro_cx_num)

    # ===== 画像を GPU に送る =====
    before_img_d = CuArray(before_img)
    after_img_d  = CuArray(after_img)

    for gy in 1:interro_cy_num
        interro_cy = interro_win_h ÷ 2 + (gy - 1) * step_y

        for gx in 1:interro_cx_num
            interro_cx = interro_win_w ÷ 2 + (gx - 1) * step_x

            # ==== 1. Interrogation Window (before_img) ==== (CPU 側と同じ)
            interro = zeros(Float64, interro_win_h, interro_win_w)
            interro_sum = 0.0

            for j in 1:interro_win_h
                for i in 1:interro_win_w
                    y = (interro_cy - interro_win_h ÷ 2) + j
                    x = (interro_cx - interro_win_w ÷ 2) + i
                    interro[j,i] = before_img[y, x]
                    interro_sum += interro[j,i]
                end
            end

            interro_mean = interro_sum / (interro_win_w * interro_win_h)

            interro_var = 0.0
            for j in 1:interro_win_h, i in 1:interro_win_w
                interro_var += (interro[j,i] - interro_mean)^2
            end
            interro_sd = sqrt(interro_var / (interro_win_w * interro_win_h))

            # ==== 2. Search Window の範囲（CPU 版と同じ）====
            ymin = interro_cy - search_h ÷ 2 + interro_win_h ÷ 2
            ymax = interro_cy + search_h ÷ 2 - interro_win_h ÷ 2
            xmin = interro_cx - search_w ÷ 2 + interro_win_w ÷ 2
            xmax = interro_cx + search_w ÷ 2 - interro_win_w ÷ 2

            # 実際の search_h, search_w（境界で多少縮む可能性も考えておくならここで更新してもよい）
            search_h_eff = ymax - ymin + 1
            search_w_eff = xmax - xmin + 1

            # ==== 3. NCC マップ R を GPU で計算 ====
            # GPU 用のバッファ
            R_d        = CuArray(zeros(Float64, search_h_eff, search_w_eff))
            interro_d  = CuArray(interro)

            cu_ncc!(
                R_d, interro_d, after_img_d,
                interro_mean, interro_sd,
                interro_win_w, interro_win_h,
                ymin, xmin,
                search_w_eff, search_h_eff,
                height, width,
            )

            # NCC マップを CPU に戻す
            R = Array(R_d)

            # ==== 4. 最大ピーク検出（CPU）====
            maxR, linear_idx = findmax(R)
            peakY, peakX = Tuple(CartesianIndices(R)[linear_idx])

            # ==== 5. サブピクセル補間（既存関数）====
            δx, δy = parabolic_subpixel(R, peakX, peakY)

            # ==== 6. 画像座標への変換 ====
            I0 = xmin
            J0 = ymin

            cand_cx_peak = (peakX + δx - 1) + I0
            cand_cy_peak = (peakY + δy - 1) + J0

            dx = cand_cx_peak - interro_cx
            dy = cand_cy_peak - interro_cy

            vec_dx[gy, gx] = dx
            vec_dy[gy, gx] = dy
            vec_R[gy, gx]  = maxR
        end
    end

    dx_filtered, dy_filtered = remove_error_vec(vec_dx, vec_dy)

    return dx_filtered, dy_filtered, vec_R
end

end # module PIVUtils
