import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os


# ============================================================
# THƯ MỤC LƯU KẾT QUẢ
# ============================================================

REPORT_DIR = "/home/thienhoang/Documents/ProcessImage/4/Report/1"
RGB_DIR = os.path.join(REPORT_DIR, "RGB")
GREY_DIR = os.path.join(REPORT_DIR, "Grey")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(GREY_DIR, exist_ok=True)


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
# KHỞI TẠO CỬA SỔ CHỌN FILE
# ============================================================

root = tk.Tk()
root.withdraw()


# ============================================================
# 1. CHỌN ẢNH MÀU
# ============================================================

print("Bước 1: Chọn ẢNH MÀU RGB.")

color_path = choose_image("Chọn ảnh MÀU RGB")

if not color_path:
    print("Bạn chưa chọn ảnh màu.")
    root.destroy()
    exit()

print("Đã chọn ảnh màu:", color_path)


# ============================================================
# 2. CHỌN ẢNH XÁM
# ============================================================

print("Bước 2: Chọn ẢNH XÁM.")

gray_path = choose_image("Chọn ảnh XÁM")

if not gray_path:
    print("Bạn chưa chọn ảnh xám.")
    root.destroy()
    exit()

print("Đã chọn ảnh xám:", gray_path)

root.destroy()


# ============================================================
# 3. ĐỌC ẢNH MÀU
# ============================================================

img_color = np.array(
    Image.open(color_path).convert("RGB")
)

R = img_color[:, :, 0]
G = img_color[:, :, 1]
B = img_color[:, :, 2]

print("\n========== ẢNH MÀU ==========")
print("Kích thước:", img_color.shape)


# ============================================================
# 4. ĐỌC ẢNH XÁM
# ============================================================

img_gray = np.array(
    Image.open(gray_path).convert("L")
)

print("\n========== ẢNH XÁM ==========")
print("Kích thước:", img_gray.shape)


# ============================================================
# LƯU ẢNH GỐC
# ============================================================

Image.fromarray(img_color).save(
    os.path.join(RGB_DIR, "Original_RGB.png")
)

Image.fromarray(img_gray).save(
    os.path.join(GREY_DIR, "Original_Gray.png")
)

print("\nĐã lưu ảnh gốc vào RGB và Grey.")


# ============================================================
# ======================= PHẦN A =============================
# HISTOGRAM ẢNH MÀU VÀ ẢNH XÁM
# ============================================================


# ------------------------------------------------------------
# A1. HIỂN THỊ 2 ẢNH
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_color)
plt.title("Ảnh màu RGB")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.title("Ảnh xám")
plt.axis("off")

plt.tight_layout()

# Lưu kết quả
plt.savefig(
    os.path.join(REPORT_DIR, "A1_Anh_Mau_Va_Xam.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ------------------------------------------------------------
# A2. HISTOGRAM ẢNH XÁM
# ------------------------------------------------------------

hist_gray, edges_gray = np.histogram(
    img_gray.ravel(),
    bins=256,
    range=(0, 256)
)

centers_gray = (
    edges_gray[:-1] + edges_gray[1:]
) / 2

plt.figure(figsize=(10, 5))

plt.bar(
    centers_gray,
    hist_gray,
    width=1
)

plt.title("Histogram - Ảnh xám")
plt.xlabel("Mức xám")
plt.ylabel("Số lượng pixel")
plt.xlim(0, 255)
plt.grid(alpha=0.3)

plt.tight_layout()

# Lưu kết quả
plt.savefig(
    os.path.join(GREY_DIR, "A2_Histogram_Anh_Xam.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ------------------------------------------------------------
# A3. HISTOGRAM ẢNH MÀU RGB
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    R.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="Red"
)

plt.hist(
    G.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="Green"
)

plt.hist(
    B.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="Blue"
)

plt.title("Histogram - Ảnh màu RGB")
plt.xlabel("Mức cường độ")
plt.ylabel("Số lượng pixel")
plt.xlim(0, 255)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# Lưu kết quả
plt.savefig(
    os.path.join(RGB_DIR, "A3_Histogram_Anh_Mau_RGB.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# ======================= PHẦN B =============================
# ẢNH HƯỞNG CỦA BINS VÀ L
# CẢ ẢNH MÀU VÀ ẢNH XÁM ĐỀU THỰC HIỆN
# ============================================================

bins_list = [8, 16, 32, 64, 128, 256]
L_list = [2, 4, 8, 16, 32, 64, 128, 256]


# ------------------------------------------------------------
# B1. ẢNH XÁM - ẢNH HƯỞNG CỦA BINS
# ------------------------------------------------------------

plt.figure(figsize=(15, 10))

for i, bins in enumerate(bins_list):

    hist, edges = np.histogram(
        img_gray.ravel(),
        bins=bins,
        range=(0, 256)
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    width = edges[1] - edges[0]

    plt.subplot(2, 3, i + 1)

    plt.bar(
        centers,
        hist,
        width=width
    )

    plt.title(f"Ảnh xám - bins = {bins}")
    plt.xlabel("Mức xám")
    plt.ylabel("Pixel")

plt.tight_layout()

# Lưu kết quả
plt.savefig(
    os.path.join(GREY_DIR, "B1_Anh_Xam_Anh_Huong_Bins.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ------------------------------------------------------------
# B2. ẢNH MÀU - ẢNH HƯỞNG CỦA BINS
# ------------------------------------------------------------

plt.figure(figsize=(15, 10))

for i, bins in enumerate(bins_list):

    plt.subplot(2, 3, i + 1)

    plt.hist(
        R.ravel(),
        bins=bins,
        range=(0, 256),
        alpha=0.5,
        label="R"
    )

    plt.hist(
        G.ravel(),
        bins=bins,
        range=(0, 256),
        alpha=0.5,
        label="G"
    )

    plt.hist(
        B.ravel(),
        bins=bins,
        range=(0, 256),
        alpha=0.5,
        label="B"
    )

    plt.title(f"Ảnh màu RGB - bins = {bins}")
    plt.xlabel("Cường độ")
    plt.ylabel("Pixel")
    plt.legend(fontsize=8)

plt.tight_layout()

# Lưu kết quả
plt.savefig(
    os.path.join(RGB_DIR, "B2_Anh_Mau_Anh_Huong_Bins.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ------------------------------------------------------------
# B3. ẢNH XÁM - ẢNH HƯỞNG CỦA L
# ------------------------------------------------------------

plt.figure(figsize=(16, 10))

for i, L in enumerate(L_list):

    # Chia 256 mức xám thành L mức
    gray_L = (
        (img_gray.astype(np.int32) * L) // 256
    )

    # Đưa mức lượng tử hóa về khoảng 0..255
    if L > 1:
        gray_L = (
            gray_L * 255 // (L - 1)
        )
    else:
        gray_L = np.zeros_like(gray_L)

    gray_L = np.clip(
        gray_L,
        0,
        255
    ).astype(np.uint8)

    # Lưu ảnh sau lượng tử hóa
    Image.fromarray(gray_L).save(
        os.path.join(
            GREY_DIR,
            f"B3_Gray_L_{L}.png"
        )
    )

    # Tính histogram
    hist, edges = np.histogram(
        gray_L.ravel(),
        bins=L,
        range=(0, 256)
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    width = edges[1] - edges[0]

    plt.subplot(2, 4, i + 1)

    plt.bar(
        centers,
        hist,
        width=width
    )

    plt.title(f"Ảnh xám - L = {L}")
    plt.xlabel("Mức xám")
    plt.ylabel("Pixel")

plt.tight_layout()

# Lưu hình tổng hợp
plt.savefig(
    os.path.join(GREY_DIR, "B3_Anh_Xam_Anh_Huong_L.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ------------------------------------------------------------
# B4. ẢNH MÀU - ẢNH HƯỞNG CỦA L
# ------------------------------------------------------------

plt.figure(figsize=(16, 10))


# Hàm lượng tử hóa
def quantize_channel(channel, levels):

    if levels == 1:
        return np.zeros_like(
            channel,
            dtype=np.uint8
        )

    q = (
        (channel.astype(np.int32) * levels) // 256
    )

    q = (
        q * 255 // (levels - 1)
    )

    return np.clip(
        q,
        0,
        255
    ).astype(np.uint8)


for i, L in enumerate(L_list):

    R_L = quantize_channel(R, L)
    G_L = quantize_channel(G, L)
    B_L = quantize_channel(B, L)

    # Ghép lại thành ảnh RGB
    img_color_L = np.stack(
        (R_L, G_L, B_L),
        axis=2
    )

    # Lưu ảnh màu sau lượng tử hóa
    Image.fromarray(img_color_L).save(
        os.path.join(
            RGB_DIR,
            f"B4_RGB_L_{L}.png"
        )
    )

    plt.subplot(2, 4, i + 1)

    plt.hist(
        R_L.ravel(),
        bins=L,
        range=(0, 256),
        alpha=0.5,
        label="R"
    )

    plt.hist(
        G_L.ravel(),
        bins=L,
        range=(0, 256),
        alpha=0.5,
        label="G"
    )

    plt.hist(
        B_L.ravel(),
        bins=L,
        range=(0, 256),
        alpha=0.5,
        label="B"
    )

    plt.title(f"Ảnh màu RGB - L = {L}")
    plt.xlabel("Cường độ")
    plt.ylabel("Pixel")
    plt.legend(fontsize=8)

plt.tight_layout()

# Lưu hình tổng hợp
plt.savefig(
    os.path.join(RGB_DIR, "B4_Anh_Mau_Anh_Huong_L.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# ======================= PHẦN C =============================
# HE - HISTOGRAM EQUALIZATION
# CHỈ THỰC HIỆN TRÊN ẢNH XÁM
# ============================================================


# ------------------------------------------------------------
# C1. TÍNH HISTOGRAM
# ------------------------------------------------------------

hist_before = np.bincount(
    img_gray.ravel(),
    minlength=256
)


# ------------------------------------------------------------
# C2. TÍNH CDF
# ------------------------------------------------------------

cdf = hist_before.cumsum()


# Giá trị CDF đầu tiên khác 0
cdf_nonzero = cdf[cdf > 0]

cdf_min = cdf_nonzero[0]


# Tổng số pixel
total_pixels = img_gray.size


# ------------------------------------------------------------
# C3. CÔNG THỨC HISTOGRAM EQUALIZATION
# ------------------------------------------------------------

if total_pixels == cdf_min:

    img_he = img_gray.copy()

else:

    mapping = (
        (cdf - cdf_min)
        / (total_pixels - cdf_min)
        * 255
    )

    mapping = np.clip(
        mapping,
        0,
        255
    ).astype(np.uint8)

    img_he = mapping[img_gray]


# ------------------------------------------------------------
# C4. HISTOGRAM SAU HE
# ------------------------------------------------------------

hist_after = np.bincount(
    img_he.ravel(),
    minlength=256
)


# ------------------------------------------------------------
# C5. LƯU ẢNH SAU HE
# ------------------------------------------------------------

Image.fromarray(img_he).save(
    os.path.join(
        GREY_DIR,
        "C5_Anh_Xam_Sau_HE.png"
    )
)


# ------------------------------------------------------------
# C6. SO SÁNH TRƯỚC VÀ SAU HE
# ------------------------------------------------------------

plt.figure(figsize=(14, 8))


# Ảnh trước HE
plt.subplot(2, 2, 1)

plt.imshow(
    img_gray,
    cmap="gray"
)

plt.title("Ảnh xám trước HE")
plt.axis("off")


# Histogram trước HE
plt.subplot(2, 2, 2)

plt.bar(
    np.arange(256),
    hist_before,
    width=1
)

plt.title("Histogram trước HE")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")
plt.xlim(0, 255)


# Ảnh sau HE
plt.subplot(2, 2, 3)

plt.imshow(
    img_he,
    cmap="gray"
)

plt.title("Ảnh xám sau HE")
plt.axis("off")


# Histogram sau HE
plt.subplot(2, 2, 4)

plt.bar(
    np.arange(256),
    hist_after,
    width=1
)

plt.title("Histogram sau HE")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")
plt.xlim(0, 255)


plt.tight_layout()

# Lưu hình so sánh
plt.savefig(
    os.path.join(
        GREY_DIR,
        "C6_So_Sanh_Truoc_Sau_HE.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


# ============================================================
# C7. IN THÔNG TIN SO SÁNH
# ============================================================

print("\n========================================")
print("THÔNG TIN ẢNH XÁM TRƯỚC VÀ SAU HE")
print("========================================")

print(f"Min trước HE : {img_gray.min()}")
print(f"Max trước HE : {img_gray.max()}")

print(f"Min sau HE   : {img_he.min()}")
print(f"Max sau HE   : {img_he.max()}")

print(f"Mean trước HE: {np.mean(img_gray):.2f}")
print(f"Mean sau HE  : {np.mean(img_he):.2f}")

print(f"Std trước HE : {np.std(img_gray):.2f}")
print(f"Std sau HE   : {np.std(img_he):.2f}")


# ============================================================
# IN THƯ MỤC LƯU
# ============================================================

print("\n========================================")
print("ĐÃ LƯU KẾT QUẢ")
print("========================================")

print("Ảnh màu và kết quả RGB:")
print(RGB_DIR)

print("\nẢnh xám và kết quả Grey:")
print(GREY_DIR)

print("\nẢnh A1:")
print(REPORT_DIR)


# ============================================================
# GIỮ TẤT CẢ CÁC FIGURE
# ============================================================

print("\nTất cả kết quả đã được mở.")
print("Đóng các cửa sổ Figure để kết thúc chương trình.")

plt.show()