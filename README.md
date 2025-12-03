# PIVUtils

**PIVUtils** は Julia で実装された **粒子画像流速測定（PIV: Particle Image Velocimetry）** 用ライブラリです。  
CPU と CUDA GPU の両方に対応しており、以下の機能を備えています。

- 正規化相互相関（NCC: Normalized Cross-Correlation）
- サブピクセル精度のピーク推定（放物線フィッティング）
- 3×3 近傍統計に基づく外れ値除去
- CPU / GPU API の統一
- ベクトル場の保存や可視化の例スクリプト

流体力学・マイクロ流体・ホログラフィック計測など、  
連続画像から速度場を推定する研究用途に向いています。

---

## 🚀 特徴

### 🔹 CPU PIV（`PIV_cpu`）
- C の古典的実装を正確に再現した NCC ベースの相関計算
- 全探索（search window scanning）
- サブピクセル補間（2 次曲線）
- 3×3 領域の統計量を用いた外れ値ベクトル除去  
  → `remove_error_vec` によるロバスト化

### 🔹 GPU PIV（`PIV_gpu`）
- CUDA kernel によって search window 全体を並列処理  
- 大規模画像で大幅な高速化
- CPU 版とほぼ同等の出力再現  
- NVIDIA GPU があれば自動的に CuArray を使用

### 🔹 補助機能
- `parabolic_subpixel(R, x0, y0)`  
  サブピクセル精度のピーク補間
- `remove_error_vec(dx, dy)`  
  近傍統計による外れ値除去
- Example スクリプトによる CPU/GPU デモンストレーション

---

### 📦 依存ライブラリ

```
CUDA
Images
StatsBase
```

---

### 🔧 開発者（PIVUtils を編集する側）

```bash
cd /path/to/PIVUtils
```

```julia
] activate .
] add CUDA Images StatsBase
```

---

### 🧑‍💻 利用者（PIVUtils を使う側）

```julia
] dev /path/to/PIVUtils
] resolve
using PIVUtils
```

---

## 📝 使い方

### CPU 版

```julia
using PIVUtils, ImageUtils

img1 = read_img_gray_float64("img_0000.png")
img2 = read_img_gray_float64("img_0001.png")

dx, dy, R = PIV_cpu(img1, img2;
    interro_win_w = 32,
    interro_win_h = 32,
    search_factor = 2
)
```

### GPU 版

```julia
using PIVUtils, ImageUtils, CUDA

img1 = read_img_gray_float64("img_0000.png")
img2 = read_img_gray_float64("img_0001.png")

dx, dy, R = PIV_gpu(img1, img2;
    interro_win_w = 32,
    interro_win_h = 32,
    search_factor = 2
)
```

---

## 📂 フォルダ構成

```
PIVUtils/
├── src/
│   └── PIVUtils.jl        # メインモジュール
├── example/
│   ├── 00_make_particles_image_cpu.jl
│   ├── 00-1_PIV_cpu.jl
│   └── 00-2_PIV_gpu.jl
├── Project.toml
├── Manifest.toml
└── README.md
```

---

## 📘 仕組みの概要

1. **interrogation window の抽出**  
2. **search window の走査**  
3. **NCC による相関計算**（CPU: ループ / GPU: CUDA kernel）
4. **ピーク検出（最大相関値）**
5. **サブピクセル補間（parabolic_subpixel）**
6. **外れ値除去（remove_error_vec）**

---

## 📄 ライセンス

MIT License.

---

## 👤 作者

Mitsuki ISHIYAMA (2025)
