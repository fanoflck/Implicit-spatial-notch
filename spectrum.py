# 对光谱进行一个展示
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# by hand
def single_image_fft(
    image_dir: str,
    save_image: bool = True,
    scale: bool = False,
    channels_merge: bool = False,
):
    im = np.asarray(Image.open(image_dir).convert("RGB"), dtype=np.float32)
    # 图像转化为 -1->1 之间的值方便进行训练

    if not channels_merge:
        for i in range(3):
            img = im[:, :, i]
            fft_img = np.fft.fft2(img)
            fft_img = np.fft.fftshift(fft_img)
            fft_img = np.log2(np.abs(fft_img))
            if scale:
                fft_min = np.percentile(fft_img, 5)
                fft_max = np.percentile(fft_img, 95)
                fft_img = (fft_img - fft_min) / (fft_max - fft_min)
                fft_img = (fft_img - 0.5) * 2
                fft_img[fft_img < -1] = -1
                fft_img[fft_img > 1] = 1
            im[:, :, i] = fft_img
        im = np.mean(im, axis=2)
    else:
        im = im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114
        im = np.fft.fft2(im)
        im = np.fft.fftshift(im)
        im = np.log(np.abs(im))

    print(im.shape)
    print(np.min(im), np.max(im))
    if save_image:
        plt.matshow(im, cmap="viridis")
        # plt.colorbar(pad=0.2)
        plt.axis("off")
        # plt.savefig(f"./resources/{image_dir.split('.')[0]}_fft2.png")
        # plt.savefig('./resources/fft2.png')
        plt.savefig("nice2.png")


# by calling libraries
def single_image_fft_(image_dir, output_dir):
    img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    )
    plt.imshow(magnitude_spectrum, cmap="viridis")
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.savefig(output_dir)


if __name__ == "__main__":
    image_dir_list = "0000fake.png"
    single_image_fft_(image_dir_list, "spct.png")
