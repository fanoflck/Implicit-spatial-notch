from utils import gaussian_noise, uniform_noise
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms, utils
import argparse
from tqdm import tqdm
from kpn import KPN
from torch.utils.tensorboard import SummaryWriter


class dataset_for_kpn(Dataset):
    def __init__(self, src_dir, mode="train") -> None:
        super().__init__()
        self.file_dir = os.path.join(src_dir, mode)
        self.files = os.listdir(self.file_dir)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_image(self, img_file):
        img = np.asarray(
            Image.open(os.path.join(self.file_dir, img_file)), dtype=np.uint8
        )
        img_real, img_fake = img[:, :128, :], img[:, 128:, :]
        if random.random() > 0.5:
            img_fake_noise = gaussian_noise(img_fake, 0, 10)
        else:
            img_fake_noise = uniform_noise(img_fake, -20, 20)

        img_real, img_fake_noise = self.transform(img_real), self.transform(
            img_fake_noise
        )
        img_fake = self.transform(img_fake)
        return img_real, img_fake, img_fake_noise

    def __getitem__(self, index):
        img_file = self.files[index]
        img_real, img_fake, img_fake_noise = self.load_image(img_file)
        return img_real, img_fake, img_fake_noise

    def __len__(self):
        return len(self.files)


def train_fn_f2r(
    epoch, loader, kpn, opt_kpn, scheduler, l1, writer, tb_step, device, path_sample
):
    loop = tqdm(loader)
    for idx, (img_real, img_fake, img_fake_noise) in enumerate(loop):

        img_real, img_fake, img_fake_noise = (
            img_real.to(device),
            img_fake.to(device),
            img_fake_noise.to(device),
        )
        recon_img = kpn(img_fake, img_fake_noise)

        # L1 loss
        loss = l1(recon_img, img_fake)

        opt_kpn.zero_grad()
        loss.backward()
        opt_kpn.step()

        writer.add_scalar(
            "loss",
            loss.item(),
            global_step=tb_step,
        )
        tb_step += 1

        loop.set_postfix(epoch=epoch, loss=loss.item())

        if idx % 100 == 0:
            kpn.eval()
            with torch.no_grad():
                recon_img = kpn(img_fake, img_fake_noise)
                img_list = [img_fake[:5], img_fake_noise[:5], recon_img[:5]]
                img_all = torch.cat(img_list, dim=0)
                utils.save_image(
                    img_all, f"{path_sample}/{epoch}_{idx}.png", normalize=True, nrow=5
                )
            kpn.train()

    scheduler.step()

    return tb_step


def main(args):

    path_sample = f"sample_{args.loss_function}"
    if not os.path.exists(path_sample):
        os.mkdir(path_sample)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    datasets = dataset_for_kpn(src_dir=args.path_data, mode="train")
    loader = DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    kpn = KPN().to(args.device)

    opt_kpn = torch.optim.Adam(kpn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt_kpn,
        T_max=args.epoch + 2,
    )

    writer = SummaryWriter("./logs_generator")
    tb_step = 0

    l1 = nn.L1Loss().to(args.device)

    kpn.train()

    for i in range(args.epoch):
        tb_step = train_fn_f2r(
            i,
            loader,
            kpn,
            opt_kpn,
            scheduler,
            l1,
            writer,
            tb_step,
            args.device,
            path_sample,
        )

        if i % 5 == 0 or i == args.epoch - 1:
            torch.save(kpn.state_dict(), "{}/dg_{}.pth".format(args.save_dir, i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--path_data", type=str, default="./resources/df_byStarGAN")
    parser.add_argument("--loss_function", type=str, default="l1")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints/checkpoints_for_kpn/try4_2multiply"
    )
    parser.add_argument("--train_way", type=str, default="f2f")
    args = parser.parse_args()

    main(args)
