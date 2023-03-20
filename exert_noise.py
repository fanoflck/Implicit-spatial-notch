from utils import gaussian_noise, uniform_noise
import os
from PIL import Image
from torchvision import transforms
from kpn import KPN
import torch
from utils import saving_image
import numpy as np
from adversarial_noise import adv_guided_noise


def noising_image(image_dir, data_type, noise_type="gaussian", filter=True, adv=True):

    print('************************************')
    print('current data,noise:{},{}'.format(data_type, noise_type))
    kpn = KPN(color=True)
    kpn.load_state_dict(torch.load('checkpoints/checkpoints_for_kpn/try_3/dg_29.pth'))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

    image_dir_all = os.path.join(image_dir, data_type)
    image_files = os.listdir(image_dir_all)

    save_path = '{}_{}'.format(noise_type, data_type)

    if adv:
        save_path = 'adv_' + save_path
    save_path = os.path.join(image_dir, save_path)

    if filter:
        save_path = save_path + "_filtered"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for index, file in enumerate(image_files):
        img_file = os.path.join(image_dir_all, file)
        image = Image.open(img_file).convert('RGB')

        if not adv:
            if noise_type == "gaussian":
                image_noise = gaussian_noise(image, mean=0, sigma=10)
            elif noise_type == "uniform":
                image_noise = uniform_noise(image, -20, 20)
            else:
                image_noise = image
        else:
            image_noise = adv_guided_noise(image)
            # adv_guided_map = adv_guided_noise(image)
            # print(np.sum(adv_guided_map))
            # if noise_type == "gaussian":
            #     image_noise = gaussian_noise(image, mean=0, sigma=13, adv_guided_map=adv_guided_map)
            # elif noise_type == "uniform":
            #     image_noise = uniform_noise(image, -26, 26, adv_guided_map)
            # else:
            #     raise AssertionError("no such type")

        if filter:
            if not adv:
                image, image_noise = transform(image), transform(image_noise)
                image, image_noise = image.unsqueeze(dim=0), image_noise.unsqueeze(dim=0)
            else:
                image = transform(image)
                image = image.unsqueeze(dim=0)

            pred_image = kpn(image, image_noise)
            saving_image(pred_image, "{}/{}".format(save_path, file))

        else:
            image_noise = Image.fromarray(image_noise)
            image_noise.save("{}/{}".format(save_path, file))

        if index % 100 == 0:
            print('{} has been completed'.format(index))


if __name__ == "__main__":
    noise_type_list = ['gaussian', 'uniform', 'None']
    data_type_list = ['celeba', 'cramergan', 'mmdgan', 'progan', 'sngan']

    for data_type in data_type_list[1:]:
        for noise_type in noise_type_list[2:]:
            noising_image('GANFingerprints/GAN_classifier_datasets/valid', data_type=data_type, noise_type=noise_type)
