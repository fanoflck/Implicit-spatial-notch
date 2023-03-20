import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class dataset_for_adv(Dataset):

    def __init__(self, root=None) -> None:
        super().__init__()
        self.root = root
        self.files = []

        real_dir = ['GANFingerprints/GAN_classifier_datasets/train/celeba']
        fake_dir = [
            'GANFingerprints/GAN_classifier_datasets/train/cramergan',
            'GANFingerprints/GAN_classifier_datasets/train/mmdgan',
            'GANFingerprints/GAN_classifier_datasets/train/progan',
            'GANFingerprints/GAN_classifier_datasets/train/sngan'
        ]

        # for files in os.listdir(real_dir[0]):
        #     self.files.append([os.path.join(real_dir[0], files), 0])

        for dir in fake_dir:
            for (index, files) in enumerate(os.listdir(dir)):
                self.files.append([os.path.join(dir, files), 1, files])

        random.shuffle(self.files)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __getitem__(self, index):
        img_file, label, _ = self.files[index]
        image = Image.open(img_file).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.int64)

        return image, label

    def __len__(self):
        return len(self.files)


def train_classifier():
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    dataset = dataset_for_adv()
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=4)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    device = "cuda:0"
    epoches = 5
    iteration = 0
    model.to(device)

    for epoch in range(epoches):
        print("Epoch {}/{}".format(epoch + 1, epoches))
        print("-" * 20)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        loop_train = tqdm(dataloader, leave=True)

        for index, (img, label) in enumerate(loop_train):

            # 前向传播
            img = img.to(device)
            label = label.to(device)
            label = label.to(torch.float32)

            optimizer.zero_grad()
            outputs = model(img)

            outputs = outputs.squeeze()
            # _, preds = torch.max(outputs.data, 1)
            preds = torch.where(outputs > 0.5, 1, 0)

            loss = criterion(outputs, label)

            loss.backward()

            # 梯度更新
            optimizer.step()

            # 记录数据
            iter_loss = loss.data.item()
            train_loss += iter_loss

            iter_corrects = torch.sum(preds == label).to(torch.float32)
            train_corrects += iter_corrects

            if index % 200 == 0:
                print("----------------------")
                print("当前的 loss为 {:.3f}".format(iter_loss))
                print("当前的 acc为 {:.3f}".format(iter_corrects / img.shape[0]))

            iteration += 1

        scheduler.step()

        if epoch % 1 == 0:
            torch.save(
                model.state_dict(),
                os.path.join("checkpoints/checkpoints_for_adv_ganfp", "Resnet50-" + str(epoch) + ".pth"),
            )


def test():
    from utils import concludescore
    from utils import drawAUC_TwoClass

    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    model.load_state_dict(torch.load('checkpoints/checkpoints_for_adv_ganfp/Resnet50-4.pth'))
    dataset = dataset_for_adv()
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=4)

    image, label = next(iter(dataloader))
    print(label)

    model.eval()  # verr important
    outputs = model(image)
    outputs = outputs.squeeze()
    print(outputs)

    predicted = torch.where(outputs > 0.5, 1, 0)
    print(predicted)

    sum_score = outputs.data.numpy()
    row = sum_score.shape
    score = []
    for i in range(row[0]):
        score.append(sum_score[i])
    # print(score)

    true_label = label.tolist()
    pre_value = predicted.tolist()
    score_result = concludescore(pre_value, true_label)
    print('准确率 精确率 召回率：\n', score_result)

    # #绘制ROC曲线
    drawAUC_TwoClass(true_label, score)


def pgd_attack(model, images, labels, eps=0.2, alpha=0.04, iters=6, device='cuda:0'):
    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.BCELoss()
    # 原图像
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        outputs = outputs.squeeze()

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        # images_grad = torch.where(images.grad > 0, 1, 0)
        adv_images = images + alpha * images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        images = torch.clamp(ori_images + eta, min=-1, max=1).detach()
    
    return images


def generate_adv_noise_pgd():
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    model.load_state_dict(torch.load('checkpoints/checkpoints_for_adv_ganfp/Resnet50-4.pth'))
    model.eval()
    device = "cuda:0"

    dataset = dataset_for_adv()
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, drop_last=False, num_workers=4)
    image, label = next(iter(dataloader))

    image = image.to(device)
    label = label.to(device)
    label = label.to(torch.float32)
    print(label)
    model = model.to(device)

    image_adv = pgd_attack(model=model, images=image, labels=label, eps=0.04, alpha=0.04, iters=5, device=device)
    perturbed = image_adv - image

    print(perturbed[0].shape)
    print(torch.count_nonzero(perturbed[0]))
    perturbed_cal = torch.where(perturbed[0] != 0, 1, 0)
    print(torch.max(perturbed_cal), torch.min(perturbed_cal), torch.sum(perturbed_cal))

    output_1 = model(image)
    print(output_1)
    output_1 = output_1.squeeze()
    predicted_1 = torch.where(output_1 > 0.5, 1, 0)

    output_2 = model(image_adv)
    print(output_2)
    output_2 = output_2.squeeze()
    predicted_2 = torch.where(output_2 > 0.5, 1, 0)

    print(label)
    print(predicted_1)
    print(predicted_2)


def adv_guided_noise(image):
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    model.load_state_dict(torch.load('checkpoints/checkpoints_for_adv_ganfp/Resnet50-4.pth'))
    model.eval()
    
    transoform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    image = transoform(image)
    image = image.unsqueeze(dim=0)
    label = torch.tensor(1, dtype=torch.float32)
    adv_image = pgd_attack(model, image, label, eps=0.2, alpha=0.04, iters=5, device="cuda:0")
    return adv_image.cpu()


if __name__ == "__main__":
    # train_classifier()
    # test()
    generate_adv_noise_pgd()

    # image = Image.open('GANFingerprints/GAN_classifier_datasets/train/cramergan/CRAMER_00000000.png')
    # adv_guided = adv_guided_noise(image)
    # print(adv_guided.shape)
