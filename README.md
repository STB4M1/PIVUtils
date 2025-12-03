# PIVUtils.jl

**PIVUtils.jl** は Julia で実装された **粒子画像流速測定（PIV: Particle Image Velocimetry）** 用ライブラリです。  
CPU と CUDA GPU の両方に対応し、次の機能を備えています：

- 正規化相互相関（NCC）
- サブピクセル精度推定（放物線フィッティング）
- 3×3 近傍統計による外れ値除去
- CPU / GPU API の統一

---

## 📦 依存ライブラリ

このパッケージが必要とする依存関係は以下の 3 つです：

```
CUDA
Images
StatsBase
```

これらは **PIVUtils のプロジェクト環境側** で管理されます。

---

## 🔧 開発者向けセットアップ（PIVUtils を開発する場合）

PIVUtils のフォルダに移動して：

```bash
cd /path/to/PIVUtils
```

### ① プロジェクト環境をアクティベート

```julia
] activate .
```

### ② 依存パッケージを追加（1回だけでOK）

```julia
] add CUDA Images StatsBase
```

これで `Project.toml` と `Manifest.toml` に依存が記録されます。

---

## 🧑‍💻 利用者向けインストール方法（ローカルパッケージとして使う）

PIVUtils は General Registry には登録されていません。  
そのため **ローカルパスを dev 登録して使用**します。

### ① PIVUtils フォルダを任意の場所に置く

例：

```
/home/user/Projects/PIVUtils
```

### ② Julia のグローバル環境を開く

```julia
julia
]
```

### ③ ローカルパッケージとして登録

```julia
dev /home/user/Projects/PIVUtils
```

相対パスでも可：

```julia
dev ./PIVUtils
```

### ④ 依存関係を解決

```julia
resolve
```

### ⑤ 使用開始

```julia
using PIVUtils
```

---

## 🚀 CPU 版の使用例

```julia
using PIVUtils, Images

img1 = Float64.(Gray.(load("img_0000.png")))
img2 = Float64.(Gray.(load("img_0001.png")))

dx, dy, R = PIV_cpu(img1, img2;
    interro_win_w = 32,
    interro_win_h = 32,
    search_factor = 2
)
```

---

## ⚡ GPU 版の使用例

```julia
using PIVUtils, Images, CUDA

img1 = Float64.(Gray.(load("img_0000.png")))
img2 = Float64.(Gray.(load("img_0001.png")))

dx, dy, R = PIV_gpu(img1, img2;
    interro_win_w = 32,
    interro_win_h = 32,
    search_factor = 2
)
```

---

## 📂 フォルダ構成（推奨）

```
PIVUtils/
├── src/
│   └── PIVUtils.jl
├── example/
│   ├── 00_make_particles_image_cpu.jl
│   ├── 00-1_PIV_cpu.jl
│   └── 00-2_PIV_gpu.jl
├── Project.toml
├── Manifest.toml
└── README.md
```

---

## 📝 ライセンス

MIT License.

---

## 👤 作者

Mitsuki ISHIYAMA (2025)
