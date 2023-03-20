from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from sklearn.metrics.pairwise import cosine_similarity as COSS
from imageio import imread
import numpy as np
from PIL import Image
from numpy import average, dot, linalg
import os


def get_thum(image, size=(128, 128), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


def measure_image_quality(src_dir, des_dir):
    metrics_list_ssim = []
    metrics_list_psnr = []
    metrics_list_coss = []
    modes = ["psnr", "ssim", "coss"]

    for index, files in enumerate(os.listdir(src_dir)):
        image_file1 = os.path.join(src_dir, files)
        image_file2 = os.path.join(des_dir, files)

        if index % 500 == 0:
            print("{} has been completed".format(index))

        for mode in modes:
            if mode == "psnr":
                image1 = imread(image_file1)
                image2 = imread(image_file2)
                result_psnr = PSNR(image1, image2)
                metrics_list_psnr.append(result_psnr)
            elif mode == "ssim":
                image1 = imread(image_file1)
                image2 = imread(image_file2)
                result_ssim = SSIM(image1, image2, multichannel=True)
                metrics_list_ssim.append(result_ssim)
            elif mode == "coss":
                image1 = Image.open(image_file1)
                image2 = Image.open(image_file2)
                result_coss = image_similarity_vectors_via_numpy(image1, image2)
                metrics_list_coss.append(result_coss)
            else:
                raise AssertionError("no such type")

    print("ssim : {}".format(np.mean(metrics_list_ssim)))
    print("psnr : {}".format(np.mean(metrics_list_psnr)))
    print("coss : {}".format(np.mean(metrics_list_coss)))


if __name__ == "__main__":
    data_type = ["progan", "sngan", "mmdgan", "cramergan"]

    for data in data_type:
        print("当前的数据类型为{}".format(data))
        src_dir = f'GANFingerprints/GAN_classifier_datasets/valid/{data}'
        des_dir = [
            f"GANFingerprints/GAN_classifier_datasets/valid/gaussian_{data}",
            f"GANFingerprints/GAN_classifier_datasets/valid/uniform_{data}",
            # f"GANFingerprints/GAN_classifier_datasets/valid/adv_None_{data}_filtered",
        ]

        for des in des_dir:
            measure_image_quality(src_dir, des)
