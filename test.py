import argparse
import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import cfg
from datasets import build_data_loader
from model import UNet3Plus, build_unet3plus
from utils.logger import AverageMeter, SummaryLogger
from utils.loss import build_u3p_loss
from utils.metrics import StreamSegMetrics


class Tester:
    loss_dict = dict()
    val_loss_dict = dict()
    val_score_dict = None
    best_val_score_dict = None

    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg_all = cfg

        # build metrics
        self.metrics = StreamSegMetrics(cfg.data.num_classes)

        cfg = self.cfg = cfg.test

        save_dir = osp.join(cfg.logger.log_dir, cfg.save_name)
        os.makedirs(save_dir, exist_ok=True)
        hyp_path = osp.join(save_dir, cfg.save_name + ".yaml")
        with open(hyp_path, "w") as f:
            f.write(cfg.dump())

        self.model: UNet3Plus = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        # build loss
        self.criterion = build_u3p_loss(cfg.loss_type, cfg.aux_weight)

        self.logger = SummaryLogger(self.cfg_all.test)

        self.model.to(cfg.device)

        # resume model
        self.resume(cfg.weight)

    def resume(self, resume_path):
        print("resuming from {}".format(resume_path))
        saved = torch.load(resume_path, map_location=self.cfg.device)
        self.model.load_state_dict(saved["state_dict"])

    def test(self):
        self.logger.info(f"start testing")
        self.test_one()

    def test_one(self):
        self.val_score_dict = self.validate()
        self.log_results()

    def update_loss_dict(self, loss_dict, batch_loss_dict=None):
        if batch_loss_dict is None:
            if loss_dict is None:
                return
            for k in loss_dict:
                loss_dict[k].reset()
        elif len(loss_dict) == 0:
            for k, v in batch_loss_dict.items():
                loss_dict[k] = AverageMeter(val=v)
        else:
            for k, v in batch_loss_dict.items():
                loss_dict[k].update(v)

    def log_results(self):
        log_dict = {"Val": {}}
        for k, v in self.val_loss_dict.items():
            log_dict["Val"][k] = v.avg
        self.update_loss_dict(self.val_loss_dict, None)

        for k, v in self.val_score_dict.items():
            if k == "Class IoU":
                print(v)
                # self.logger.cmd_logger.info(v)
                continue
            log_dict["Val"][k] = v
        self.logger.summary(log_dict, 0)

    def validate(self):
        """Do validation and return specified samples"""
        self.metrics.reset()
        self.model.eval()
        device = self.cfg.device
        pbar = enumerate(self.val_loader)
        pbar = tqdm(pbar, total=len(self.val_loader), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
        with torch.no_grad():
            for i, (images, labels) in pbar:
                image_filename = [os.path.basename(f) for f in self.val_loader.dataset.images]
                if i == 810 // self.val_loader.batch_size:
                    image = Image.open(self.val_loader.dataset.images[810]).convert("RGB")
                    image.save(os.path.basename(self.val_loader.dataset.images[810]))
                images = images.to(device)

                labels = labels.to(device, dtype=torch.long)

                outputs = self.model(images)

                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                if i == 810 // self.val_loader.batch_size:
                    cmap = plt.get_cmap("nipy_spectral", 22)
                    image = cmap(preds[810 % self.val_loader.batch_size])
                    plt.imshow(image)
                    filename, extension = os.path.splitext(os.path.basename(self.val_loader.dataset.images[810]))
                    plt.savefig(filename + "_preds" + extension)

                    image = cmap(targets[810 % self.val_loader.batch_size])
                    plt.imshow(image)
                    plt.savefig(filename + "_label" + extension)
                self.metrics.update(targets, preds)

                _, batch_loss_dict = self.criterion(outputs, labels)
                self.update_loss_dict(self.val_loss_dict, batch_loss_dict)

            score = self.metrics.get_results()
            pbar.close()
        return score


def main(args):
    cfg.merge_from_file(args.cfg)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.weight:
        cfg.test.weight = args.weight
    cfg.freeze()
    print(cfg)

    import random

    import numpy as np
    import torch

    torch.cuda.set_per_process_memory_fraction(cfg.model.memory)

    torch.cuda.manual_seed_all(cfg.test.seed)
    torch.manual_seed(cfg.test.seed)
    random.seed(cfg.test.seed)
    np.random.seed(cfg.test.seed)

    model, data = cfg.model, cfg.data
    model = build_unet3plus(
        data.num_classes, model.encoder, model.skip_ch, model.aux_losses, model.use_cgm, model.pretrained, model.dropout
    )

    if data.type in ["voc2012", "voc2012_aug"]:
        train_loader, val_loader = build_data_loader(
            data.data_dir, data.batch_size, data.num_workers, data.max_training_samples, data.crop_size
        )
    else:
        raise NotImplementedError

    tester = Tester(cfg, model, train_loader, val_loader)
    tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test segmentation network")

    parser.add_argument("--cfg", help="experiment configure file name", default="config/resnet34_voc.yaml", type=str)
    parser.add_argument("--seed", help="random seed", default=None)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--weight", default=None, type=str)

    args = parser.parse_args()
    main(args)
