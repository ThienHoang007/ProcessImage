import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os


# ============================================================
# THƯ MỤC LƯU KẾT QUẢ
# ============================================================

REPORT_DIR = "/home/thienhoang/Documents/ProcessImage/4/Report/3"

SOBEL_DIR = os.path.join(REPORT_DIR, "Sobel")
LAPLACE_DIR = os.path.join(REPORT_DIR, "Laplace")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SOBEL_DIR, exist_ok=True)
os.makedirs(LAPLACE_DIR, exist_ok=True)


# ============================================================
# HÀM CHỌN ẢNH
# ============================================================

def choose_image(title):

    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )

    return file_path


# ============================================================
# HÀM CHUYỂN ẢNH SANG ẢNH XÁM
# Không dùng OpenCV
# ============================================================

def rgb_to_gray(image):

    if image.ndim == 2:
        return image.astype(np.uint8)

    R = image[:, :, 0].astype(np.float64)
    G = image[:, :, 1].astype(np.float64)
    B = image[:, :, 2].astype(np.float64)

    gray = (
        0.299 * R
        + 0.587 * G
        + 0.114 * B
    )

    return np.clip(
        gray,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# HÀM CONVOLUTION 2D
# TỰ CÀI ĐẶT
# ============================================================

def convolution2d(image, kernel):

    image = image.astype(np.float64)

    kernel = np.array(
        kernel,
        dtype=np.float64
    )

    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        (
            (pad_h, pad_h),
            (pad_w, pad_w)
        ),
        mode="edge"
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kh, kw)
    )

    # Đảo kernel theo đúng phép convolution
    kernel_flipped = np.flip(
        kernel,
        axis=(0, 1)
    )

    result = np.sum(
        windows * kernel_flipped,
        axis=(-2, -1)
    )

    return result


# ============================================================
# HÀM CHUẨN HÓA ẢNH KẾT QUẢ
# ============================================================

def normalize_image(image):

    image = np.abs(image)

    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    result = (
        (image - min_value)
        / (max_value - min_value)
        * 255
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# KHỞI TẠO TKINTER
# ============================================================

root = tk.Tk()
root.withdraw()


# ============================================================
# 1. CHỌN ẢNH
# ============================================================

print("========================================")
print("CHỌN ẢNH ĐỂ TÁCH CẠNH")
print("========================================")

image_path = choose_image(
    "Chọn ảnh cần tách cạnh"
)

if not image_path:

    print("Bạn chưa chọn ảnh.")

    root.destroy()

    exit()

print("Ảnh đã chọn:")
print(image_path)

root.destroy()


# ============================================================
# 2. ĐỌC ẢNH
# ============================================================

img_color = np.array(
    Image.open(image_path).convert("RGB")
)

img_gray = rgb_to_gray(
    img_color
)

print("\n========================================")
print("THÔNG TIN ẢNH")
print("========================================")

print("Kích thước ảnh màu:", img_color.shape)
print("Kích thước ảnh xám :", img_gray.shape)


# ============================================================
# LƯU ẢNH GỐC
# ============================================================

Image.fromarray(
    img_color
).save(
    os.path.join(
        REPORT_DIR,
        "Original_RGB.png"
    )
)

Image.fromarray(
    img_gray
).save(
    os.path.join(
        REPORT_DIR,
        "Original_Gray.png"
    )
)


# ============================================================
# ======================= SOBEL ==============================
# ============================================================

print("\n========================================")
print("PHẦN 1: SOBEL")
print("========================================")


# ------------------------------------------------------------
# SOBEL GX
# ------------------------------------------------------------

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)


# ------------------------------------------------------------
# SOBEL GY
# ------------------------------------------------------------

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)


# ------------------------------------------------------------
# TÍNH GRADIENT
# ------------------------------------------------------------

Gx = convolution2d(
    img_gray,
    sobel_x
)

Gy = convolution2d(
    img_gray,
    sobel_y
)


# Độ lớn gradient
gradient_magnitude = np.sqrt(
    Gx ** 2 + Gy ** 2
)

sobel_result = normalize_image(
    gradient_magnitude
)


# ------------------------------------------------------------
# LƯU GX
# ------------------------------------------------------------

Gx_image = normalize_image(Gx)

Image.fromarray(
    Gx_image
).save(
    os.path.join(
        SOBEL_DIR,
        "Sobel_Gx.png"
    )
)


# ------------------------------------------------------------
# LƯU GY
# ------------------------------------------------------------

Gy_image = normalize_image(Gy)

Image.fromarray(
    Gy_image
).save(
    os.path.join(
        SOBEL_DIR,
        "Sobel_Gy.png"
    )
)


# ------------------------------------------------------------
# LƯU GRADIENT
# ------------------------------------------------------------

Image.fromarray(
    sobel_result
).save(
    os.path.join(
        SOBEL_DIR,
        "Sobel_Gradient.png"
    )
)


# ============================================================
# SOBEL THAY ĐỔI TRỌNG SỐ
# ============================================================

# Mặt nạ Sobel thông thường
sobel_x_normal = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

sobel_y_normal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)


# Sobel với trọng số trung tâm lớn hơn
sobel_x_weighted = np.array([
    [-1, 0, 1],
    [-3, 0, 3],
    [-1, 0, 1]
], dtype=np.float64)

sobel_y_weighted = np.array([
    [-1, -3, -1],
    [ 0,  0,  0],
    [ 1,  3,  1]
], dtype=np.float64)


# Tính kết quả Sobel có trọng số
Gx_weighted = convolution2d(
    img_gray,
    sobel_x_weighted
)

Gy_weighted = convolution2d(
    img_gray,
    sobel_y_weighted
)

gradient_weighted = np.sqrt(
    Gx_weighted ** 2
    + Gy_weighted ** 2
)

sobel_weighted_result = normalize_image(
    gradient_weighted
)


Image.fromarray(
    sobel_weighted_result
).save(
    os.path.join(
        SOBEL_DIR,
        "Sobel_Weighted.png"
    )
)


# ============================================================
# HIỂN THỊ SO SÁNH SOBEL
# ============================================================

plt.figure(figsize=(16, 10))


plt.subplot(2, 3, 1)

plt.imshow(
    img_gray,
    cmap="gray"
)

plt.title("Ảnh xám")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    Gx_image,
    cmap="gray"
)

plt.title("Sobel Gx")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    Gy_image,
    cmap="gray"
)

plt.title("Sobel Gy")
plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    sobel_result,
    cmap="gray"
)

plt.title("Sobel chuẩn")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    sobel_weighted_result,
    cmap="gray"
)

plt.title("Sobel thay đổi trọng số")
plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    gradient_magnitude,
    cmap="gray"
)

plt.title("Gradient magnitude")
plt.axis("off")


plt.tight_layout()


plt.savefig(
    os.path.join(
        SOBEL_DIR,
        "So_Sanh_Sobel.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# IN MẶT NẠ SOBEL
# ============================================================

print("\n----------------------------------------")
print("MẶT NẠ SOBEL CHUẨN")
print("----------------------------------------")

print("Sobel Gx:")
print(sobel_x)

print("\nSobel Gy:")
print(sobel_y)


print("\n----------------------------------------")
print("MẶT NẠ SOBEL THAY ĐỔI TRỌNG SỐ")
print("----------------------------------------")

print("Sobel Gx weighted:")
print(sobel_x_weighted)

print("\nSobel Gy weighted:")
print(sobel_y_weighted)


# ============================================================
# ======================= LAPLACE ============================
# ============================================================

print("\n========================================")
print("PHẦN 2: LAPLACE")
print("========================================")


# ------------------------------------------------------------
# LAPLACE 4 LÁNG GIỀNG
# ------------------------------------------------------------

laplace_4 = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float64)


# ------------------------------------------------------------
# LAPLACE 8 LÁNG GIỀNG
# ------------------------------------------------------------

laplace_8 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)


# ------------------------------------------------------------
# LAPLACE TRỌNG SỐ KHÁC
# ------------------------------------------------------------

laplace_weighted = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float64)


# ------------------------------------------------------------
# TÍNH LAPLACE
# ------------------------------------------------------------

laplace_result_4 = convolution2d(
    img_gray,
    laplace_4
)

laplace_result_8 = convolution2d(
    img_gray,
    laplace_8
)

laplace_result_weighted = convolution2d(
    img_gray,
    laplace_weighted
)


# Chuẩn hóa
laplace_4_image = normalize_image(
    laplace_result_4
)

laplace_8_image = normalize_image(
    laplace_result_8
)

laplace_weighted_image = normalize_image(
    laplace_result_weighted
)


# ============================================================
# LƯU KẾT QUẢ LAPLACE
# ============================================================

Image.fromarray(
    laplace_4_image
).save(
    os.path.join(
        LAPLACE_DIR,
        "Laplace_4_Neighbor.png"
    )
)

Image.fromarray(
    laplace_8_image
).save(
    os.path.join(
        LAPLACE_DIR,
        "Laplace_8_Neighbor.png"
    )
)

Image.fromarray(
    laplace_weighted_image
).save(
    os.path.join(
        LAPLACE_DIR,
        "Laplace_Weighted.png"
    )
)


# ============================================================
# HIỂN THỊ SO SÁNH LAPLACE
# ============================================================

plt.figure(figsize=(14, 8))


plt.subplot(2, 2, 1)

plt.imshow(
    img_gray,
    cmap="gray"
)

plt.title("Ảnh xám")
plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    laplace_4_image,
    cmap="gray"
)

plt.title("Laplace - 4 láng giềng")
plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    laplace_8_image,
    cmap="gray"
)

plt.title("Laplace - 8 láng giềng")
plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    laplace_weighted_image,
    cmap="gray"
)

plt.title("Laplace - thay đổi trọng số")
plt.axis("off")


plt.tight_layout()


plt.savefig(
    os.path.join(
        LAPLACE_DIR,
        "So_Sanh_Laplace.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# IN MẶT NẠ LAPLACE
# ============================================================

print("\n----------------------------------------")
print("MẶT NẠ LAPLACE 4 LÁNG GIỀNG")
print("----------------------------------------")

print(laplace_4)


print("\n----------------------------------------")
print("MẶT NẠ LAPLACE 8 LÁNG GIỀNG")
print("----------------------------------------")

print(laplace_8)


print("\n----------------------------------------")
print("MẶT NẠ LAPLACE THAY ĐỔI TRỌNG SỐ")
print("----------------------------------------")

print(laplace_weighted)


# ============================================================
# ===================== SO SÁNH CHUNG ========================
# ============================================================

plt.figure(figsize=(16, 8))


plt.subplot(2, 3, 1)

plt.imshow(
    img_gray,
    cmap="gray"
)

plt.title("Ảnh gốc xám")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    sobel_result,
    cmap="gray"
)

plt.title("Sobel")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    sobel_weighted_result,
    cmap="gray"
)

plt.title("Sobel có trọng số")
plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    laplace_4_image,
    cmap="gray"
)

plt.title("Laplace 4")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    laplace_8_image,
    cmap="gray"
)

plt.title("Laplace 8")
plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    laplace_weighted_image,
    cmap="gray"
)

plt.title("Laplace có trọng số")
plt.axis("off")


plt.tight_layout()


plt.savefig(
    os.path.join(
        REPORT_DIR,
        "So_Sanh_Sobel_Laplace.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# IN THÔNG TIN
# ============================================================

print("\n========================================")
print("HOÀN THÀNH BÀI 3")
print("========================================")

print("\nKết quả Sobel:")
print(SOBEL_DIR)

print("\nKết quả Laplace:")
print(LAPLACE_DIR)

print("\nKết quả so sánh chung:")
print(REPORT_DIR)


# ============================================================
# GIỮ CÁC CỬA SỔ FIGURE
# ============================================================

print("\nĐóng các cửa sổ Figure để kết thúc chương trình.")

plt.show()
