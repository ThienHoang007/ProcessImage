import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os


# ============================================================
# THƯ MỤC LƯU KẾT QUẢ
# ============================================================

REPORT_DIR = "/home/thienhoang/Documents/ProcessImage/4/Report/2"

MEDIAN_DIR = os.path.join(REPORT_DIR, "Median")
GAUSSIAN_DIR = os.path.join(REPORT_DIR, "Gaussian")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MEDIAN_DIR, exist_ok=True)
os.makedirs(GAUSSIAN_DIR, exist_ok=True)


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
# HÀM MEDIAN FILTER
# Không sử dụng OpenCV
# ============================================================

def median_filter(image, kernel_size=3):

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size phải là số lẻ.")

    pad = kernel_size // 2

    # Ảnh xám
    if image.ndim == 2:

        padded = np.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode="edge"
        )

        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (kernel_size, kernel_size)
        )

        result = np.median(
            windows,
            axis=(-2, -1)
        )

        return result.astype(np.uint8)

    # Ảnh màu
    elif image.ndim == 3:

        channels = []

        for c in range(image.shape[2]):

            channel = image[:, :, c]

            padded = np.pad(
                channel,
                ((pad, pad), (pad, pad)),
                mode="edge"
            )

            windows = np.lib.stride_tricks.sliding_window_view(
                padded,
                (kernel_size, kernel_size)
            )

            filtered = np.median(
                windows,
                axis=(-2, -1)
            )

            channels.append(
                filtered.astype(np.uint8)
            )

        return np.stack(channels, axis=2)

    else:
        raise ValueError("Định dạng ảnh không hợp lệ.")


# ============================================================
# TẠO GAUSSIAN KERNEL
# ============================================================

def gaussian_kernel(kernel_size=5, sigma=1.0):

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size phải là số lẻ.")

    pad = kernel_size // 2

    x = np.arange(-pad, pad + 1)

    y = np.arange(-pad, pad + 1)

    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    # Chuẩn hóa tổng kernel = 1
    kernel = kernel / np.sum(kernel)

    return kernel


# ============================================================
# HÀM GAUSSIAN FILTER
# Không sử dụng OpenCV
# ============================================================

def gaussian_filter(image, kernel_size=5, sigma=1.0):

    kernel = gaussian_kernel(
        kernel_size,
        sigma
    )

    pad = kernel_size // 2

    # --------------------------------------------------------
    # ẢNH XÁM
    # --------------------------------------------------------

    if image.ndim == 2:

        padded = np.pad(
            image.astype(np.float64),
            ((pad, pad), (pad, pad)),
            mode="edge"
        )

        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (kernel_size, kernel_size)
        )

        result = np.sum(
            windows * kernel,
            axis=(-2, -1)
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    # --------------------------------------------------------
    # ẢNH MÀU
    # --------------------------------------------------------

    elif image.ndim == 3:

        channels = []

        for c in range(image.shape[2]):

            channel = image[:, :, c]

            padded = np.pad(
                channel.astype(np.float64),
                ((pad, pad), (pad, pad)),
                mode="edge"
            )

            windows = np.lib.stride_tricks.sliding_window_view(
                padded,
                (kernel_size, kernel_size)
            )

            filtered = np.sum(
                windows * kernel,
                axis=(-2, -1)
            )

            filtered = np.clip(
                filtered,
                0,
                255
            ).astype(np.uint8)

            channels.append(filtered)

        return np.stack(channels, axis=2)

    else:
        raise ValueError("Định dạng ảnh không hợp lệ.")


# ============================================================
# HÀM HIỂN THỊ ẢNH
# ============================================================

def show_image(image, title):

    if image.ndim == 2:

        plt.imshow(
            image,
            cmap="gray"
        )

    else:

        plt.imshow(image)

    plt.title(title)
    plt.axis("off")


# ============================================================
# KHỞI TẠO TKINTER
# ============================================================

root = tk.Tk()
root.withdraw()


# ============================================================
# 1. CHỌN ẢNH NHIỄU MUỐI TIÊU
# ============================================================

print("========================================")
print("BƯỚC 1: CHỌN ẢNH NHIỄU MUỐI TIÊU")
print("========================================")

salt_pepper_path = choose_image(
    "Chọn ảnh chứa nhiễu Muối tiêu"
)

if not salt_pepper_path:

    print("Bạn chưa chọn ảnh nhiễu Muối tiêu.")

    root.destroy()

    exit()

print(
    "Ảnh nhiễu Muối tiêu:",
    salt_pepper_path
)


# ============================================================
# 2. CHỌN ẢNH NHIỄU GAUSS
# ============================================================

print("\n========================================")
print("BƯỚC 2: CHỌN ẢNH NHIỄU GAUSS")
print("========================================")

gaussian_path = choose_image(
    "Chọn ảnh chứa nhiễu Gauss"
)

if not gaussian_path:

    print("Bạn chưa chọn ảnh nhiễu Gauss.")

    root.destroy()

    exit()

print(
    "Ảnh nhiễu Gauss:",
    gaussian_path
)

root.destroy()


# ============================================================
# 3. ĐỌC ẢNH BẰNG PILLOW
# ============================================================

img_sp = np.array(
    Image.open(salt_pepper_path).convert("RGB")
)

img_gaussian = np.array(
    Image.open(gaussian_path).convert("RGB")
)


print("\n========================================")
print("THÔNG TIN ẢNH")
print("========================================")

print("Ảnh Muối tiêu:")
print("Kích thước:", img_sp.shape)

print("\nẢnh Gauss:")
print("Kích thước:", img_gaussian.shape)


# ============================================================
# 4. LƯU ẢNH GỐC
# ============================================================

Image.fromarray(img_sp).save(
    os.path.join(
        MEDIAN_DIR,
        "Original_Salt_Pepper.png"
    )
)

Image.fromarray(img_gaussian).save(
    os.path.join(
        GAUSSIAN_DIR,
        "Original_Gaussian.png"
    )
)


# ============================================================
# 5. LỌC MEDIAN
# ============================================================

print("\n========================================")
print("ĐANG THỰC HIỆN MEDIAN FILTER")
print("========================================")

median_result = median_filter(
    img_sp,
    kernel_size=3
)


# ============================================================
# 6. LỌC GAUSSIAN
# ============================================================

print("\n========================================")
print("ĐANG THỰC HIỆN GAUSSIAN FILTER")
print("========================================")

gaussian_result = gaussian_filter(
    img_gaussian,
    kernel_size=5,
    sigma=1.0
)


# ============================================================
# 7. LƯU ẢNH KẾT QUẢ
# ============================================================

median_output = os.path.join(
    MEDIAN_DIR,
    "Median_Filter_Result.png"
)

gaussian_output = os.path.join(
    GAUSSIAN_DIR,
    "Gaussian_Filter_Result.png"
)

Image.fromarray(
    median_result
).save(median_output)

Image.fromarray(
    gaussian_result
).save(gaussian_output)


# ============================================================
# 8. HIỂN THỊ VÀ LƯU SO SÁNH MEDIAN
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

show_image(
    img_sp,
    "Ảnh nhiễu Muối tiêu"
)

plt.subplot(1, 2, 2)

show_image(
    median_result,
    "Sau lọc Median 3x3"
)

plt.tight_layout()

median_compare = os.path.join(
    MEDIAN_DIR,
    "So_Sanh_Median.png"
)

plt.savefig(
    median_compare,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# 9. HIỂN THỊ VÀ LƯU SO SÁNH GAUSSIAN
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

show_image(
    img_gaussian,
    "Ảnh nhiễu Gauss"
)

plt.subplot(1, 2, 2)

show_image(
    gaussian_result,
    "Sau lọc Gaussian 5x5"
)

plt.tight_layout()

gaussian_compare = os.path.join(
    GAUSSIAN_DIR,
    "So_Sanh_Gaussian.png"
)

plt.savefig(
    gaussian_compare,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# 10. IN MA TRẬN GAUSSIAN KERNEL
# ============================================================

kernel = gaussian_kernel(
    kernel_size=5,
    sigma=1.0
)

print("\n========================================")
print("GAUSSIAN KERNEL 5x5")
print("========================================")

print(kernel)


# ============================================================
# 11. IN THÔNG TIN KẾT QUẢ
# ============================================================

print("\n========================================")
print("ĐÃ HOÀN THÀNH")
print("========================================")

print("\nMedian Filter:")
print(
    "Ảnh kết quả:",
    median_output
)

print(
    "Hình so sánh:",
    median_compare
)

print("\nGaussian Filter:")
print(
    "Ảnh kết quả:",
    gaussian_output
)

print(
    "Hình so sánh:",
    gaussian_compare
)


# ============================================================
# 12. GIỮ CÁC CỬA SỔ
# ============================================================

print("\nĐóng các cửa sổ Figure để kết thúc chương trình.")

plt.show()