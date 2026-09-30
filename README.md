# GAN cho dữ liệu mất cân bằng — Tiền xử lý dữ liệu theo DeepSMOTE

Bước này chỉ **chuẩn bị dữ liệu** theo protocol thực nghiệm của paper DeepSMOTE
([arXiv:2105.02340](https://arxiv.org/abs/2105.02340)). Bước này **không** train autoencoder, không chạy
SMOTE/DeepSMOTE, không train GAN hay classifier và không tính metric nào.

Với mỗi dataset:

1. Load train split và test split chính thức gốc (tải vào `data/raw/`, không bao giờ bị sửa).
2. Tách tập **train** theo class và chọn ngẫu nhiên đúng số mẫu mỗi class như paper: sampling đều
   **không hoàn lại**, dùng một `numpy.random.Generator(seed)` duy nhất. Không tạo mẫu tổng hợp,
   không trùng lặp.
3. Giữ nguyên tập **test**.
4. Lưu tất cả, sau đó load lại từ đĩa và kiểm tra (validation).

## Cấu hình mất cân bằng (`src/preprocessing/config.py`)

| Dataset | Số mẫu train mỗi class (class 0 → 9 / 0 → 4) | Tổng | Tỉ lệ |
|---|---|---|---|
| MNIST, Fashion-MNIST | 4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40 | 9.000 | 100:1 |
| CIFAR-10, SVHN | 4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80 | 10.280 | 56,25:1 |
| CelebA (màu tóc) | 9000, 4500, 1000, 500, 160 | 15.160 | 56,25:1 |

Các class của CelebA: `0 Black_Hair, 1 Brown_Hair, 2 Blond_Hair, 3 Gray_Hair, 4 Bald`. Paper chỉ ghi
"5 classes … resized to 3×32×32", nên đây là các giả định của chúng tôi:
- chỉ ảnh có **đúng một** trong năm thuộc tính này mới được gán nhãn; ảnh mơ hồ không bao giờ được sample;
- train = partition train chính thức của CelebA, test = partition test chính thức, chỉ giữ ảnh có nhãn
  (giữ toàn bộ);
- ảnh được center-crop về 178×178 rồi resize về 32×32 (Lanczos).

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # hoặc bản CUDA
pip install -r requirements.txt
```

`scipy` chỉ cần cho SVHN, `gdown` chỉ cần để tự động tải CelebA.

## Cách chạy

```bash
python src/preprocessing/create_imbalanced_dataset.py --dataset mnist --seed 42
python src/preprocessing/create_imbalanced_dataset.py --dataset fashion_mnist --seed 42
python src/preprocessing/create_imbalanced_dataset.py --dataset cifar10 --seed 42
python src/preprocessing/create_imbalanced_dataset.py --dataset svhn --seed 42
python src/preprocessing/create_imbalanced_dataset.py --dataset celeba --seed 42
python src/preprocessing/create_imbalanced_dataset.py --dataset all --seed 42
```

Tùy chọn:

| Flag | Ý nghĩa |
|---|---|
| `--seed` | random seed (mặc định 42) |
| `--raw-dir` | thư mục chứa dataset gốc (mặc định `data/raw`) |
| `--output-dir` | thư mục gốc cho output đã xử lý (mặc định `data/processed`) |
| `--no-download` | báo lỗi thay vì tải dữ liệu còn thiếu |
| `--export-png` | ghi thêm PNG + `dataset.json` cho StyleGAN2-ADA |

Exit code khác 0 nếu có dataset không đạt validation.

**Tải CelebA:** torchvision tải CelebA từ Google Drive, nơi thường bị giới hạn lượt tải. Nếu lỗi, hãy đặt
thư mục `img_align_celeba/` (đã giải nén) cùng các file `list_*.txt` / `identity_CelebA.txt` vào
`data/raw/celeba/celeba/` rồi chạy lại.

## Cấu trúc output

```
data/
├── raw/<dataset>/                          # dữ liệu gốc đã tải, không bị động tới
└── processed/<dataset>/imbalance_<tỉ lệ>/seed_<seed>/
    ├── train.npz        # x: uint8 (N,H,W,C)  y: int64 (N,)  source_indices: int64 (N,)
    ├── test.npz         # x, y: test split gốc, giữ nguyên
    ├── metadata.json    # seed, số mẫu mỗi class, tỉ lệ, hash, kết quả validation
    └── stylegan2_ada/   # chỉ có khi dùng --export-png: <class>/imgXXXXXX.png + dataset.json
```

Ví dụ: `data/processed/mnist/imbalance_100/seed_42/`. Mỗi seed có thư mục riêng, nên chạy nhiều lần với
các seed khác nhau (ví dụ cho nhiều fold hoặc nhiều lần lặp) sẽ không ghi đè lên nhau.

- Ảnh giữ nguyên giá trị pixel `uint8` trong `[0, 255]`, layout NHWC. Việc chuẩn hóa (ví dụ về `[-1, 1]`
  cho decoder/generator dùng tanh) thuộc về code model.
- `source_indices[i]` là vị trí của mẫu train `i` trong train split chính thức gốc, nên mọi tập con đều
  truy ngược được về dữ liệu gốc.
- Các mẫu được nhóm theo class (0, 1, …) và sắp xếp theo source index trong mỗi class. Hãy shuffle trong
  DataLoader.

Load dữ liệu:

```python
import numpy as np
train = np.load("data/processed/mnist/imbalance_100/seed_42/train.npz")
x_train, y_train = train["x"], train["y"]
test = np.load("data/processed/mnist/imbalance_100/seed_42/test.npz")
```

StyleGAN2-ADA (conditional):

```bash
python src/preprocessing/create_imbalanced_dataset.py --dataset cifar10 --export-png
python dataset_tool.py --source=data/processed/cifar10/imbalance_56.25/seed_42/stylegan2_ada --dest=cifar10_imb.zip
# MNIST/Fashion-MNIST có kích thước 28x28: thêm --width=32 --height=32 (StyleGAN2-ADA cần kích thước lũy thừa của 2)
```

## Validation

Sau khi lưu, script load lại các file `.npz` và kiểm tra:

| Kiểm tra | Nội dung |
|---|---|
| `class_distribution` | số mẫu mỗi class bằng cấu hình; mọi nhãn đều là class hợp lệ |
| `total_samples` | tổng = tổng trong cấu hình; kích thước ảnh và dtype đúng |
| `no_duplicate_indices` | không có source index nào lặp lại, trên toàn tập hay trong từng class |
| `labels_match_original` | nhãn đã lưu bằng nhãn gốc tại `source_indices` |
| `images_are_original` | ảnh đã lưu giống hệt từng byte với ảnh gốc (không có ảnh tổng hợp hay bị sửa) |
| `seed_reproducible` | sample lại với cùng seed cho index giống hệt; seed+1 cho kết quả khác |
| `test_set_unchanged` | test set đã lưu giống hệt bản gốc (và đúng kích thước chính thức) |

Kết quả được in ra và lưu trong `metadata.json` cùng với hash SHA-256 của các index đã sample và của ảnh,
để có thể so sánh các lần chạy trên những máy khác nhau.

## Liên hệ với protocol của paper

Paper báo cáo kết quả với 5-fold cross-validation, dùng cả test set mất cân bằng và test set cân bằng.
Theo phạm vi của bước này, chúng tôi chỉ tạo **training set mất cân bằng + test set gốc**. Các protocol bổ
sung (fold, tập con test cân bằng) có thể thêm sau dựa trên `sample_imbalanced_indices` trong
`src/preprocessing/sampling.py`, mà không làm thay đổi dữ liệu train đã tạo ở đây.

Code gốc của tác giả trong `DeepSMOTE/` đọc file text float đã flatten (`np.loadtxt`, shape `(N, 784)`).
Để dùng dữ liệu này với code đó sau này: `np.savetxt(path, train["x"].reshape(len(train["x"]), -1))` và
`np.savetxt(path, train["y"])`.

## Code

```
src/preprocessing/
├── config.py                     # DATASET_CONFIGS: số mẫu mỗi class, tên class, đường dẫn, seed
├── loaders.py                    # train/test split gốc qua torchvision (uint8 NHWC)
├── sampling.py                   # sampling theo class có seed, không hoàn lại
├── validation.py                 # các bước kiểm tra ở trên
└── create_imbalanced_dataset.py  # CLI: sample → lưu → load lại → validate → metadata
```

Dùng từ một notebook ở thư mục gốc của project:

```python
import sys; sys.path.insert(0, "src")
from preprocessing.create_imbalanced_dataset import create_imbalanced_dataset
create_imbalanced_dataset("mnist", seed=42)
```
