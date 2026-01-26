import gc
import os
import glob
import argparse

import rasterio
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
import kornia.augmentation as K
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassConfusionMatrix
from torchgeo.trainers import SemanticSegmentationTask

from dataloaders.dfc2022_datamodule import DFC2022DataModule
from dataloaders.isprs_datamodule import ISPRSDataModule


@torch.no_grad()
def predict_torch_metrics(config_file, log_dir, device):
    print('using ', device)
    general_config = OmegaConf.load(config_file)

    # Load checkpoint and config
    trained_params = OmegaConf.load(os.path.join(log_dir, "hparams.yaml"))
    ckpt = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))[0]

    # Load model
    task = SemanticSegmentationTask.load_from_checkpoint(ckpt)
    task = task.to(device)  # .to(torch.float16)
    task.eval()

    # Load datamodule and dataloader
    if general_config.datamodule.dataset == 'DFC2022':
        datamodule = DFC2022DataModule(**general_config.datamodule)
        datamodule.setup()
        dataloader = datamodule.test_dataloader()  # batch size is 1

        pad = K.PadTo(size=(2048, 2048), pad_mode="constant", pad_value=0.0)

        indices = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14]).int().to(device)  # 12 classes
    else:  # vaihingen and potsdam
        datamodule = ISPRSDataModule(**general_config.datamodule)
        datamodule.setup()
        dataloader = datamodule.predict_dataloader()

        indices = torch.Tensor([0, 1, 2, 3, 4, 5]).int().to(device)  # 6 classes

    jaccard = torch.zeros(trained_params.num_classes).to(device)
    jaccard_metric = MulticlassJaccardIndex(num_classes=trained_params.num_classes,
                                            ignore_index=trained_params.ignore_index, average="none").to(device)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = batch["image"].to(device)  # .to(torch.float16)
        print('input image', x.shape, batch["mask"].shape)
        if general_config.datamodule.dataset == 'DFC2022':
            h, w = x.shape[-2:]
            x = pad(x)
            preds = task(x)
            preds = preds[0, :, :h, :w]
            print('preds', preds.shape)
        else:
            prob_im = torch.zeros((general_config.learning.num_classes, batch["mask"].shape[1],
                                   batch["mask"].shape[2]), dtype=torch.float).to(device)
            occur_im = torch.zeros((general_config.learning.num_classes,
                                    batch["mask"].shape[1], batch["mask"].shape[2]), dtype=torch.float).to(device)
            bs = general_config.datamodule.batch_size
            for j in range(len(x)//bs + 1):
                preds_patch = task(x[bs*j:min(bs*j+bs, len(x)), :, :, :])
                for k, (cur_x, cur_y) in enumerate(batch["coords"][bs*j:min(bs*j+bs, len(x))]):
                    prob_im[:, cur_x:cur_x + 256, cur_y:cur_y + 256] += preds_patch[k, :, :, :]
                    occur_im[:, cur_x:cur_x + 256, cur_y:cur_y + 256] += 1
            # print('sanity check', torch.sum(prob_im == 0), torch.sum(occur_im == 0))
            occur_im[occur_im == 0] = 1
            preds = prob_im / occur_im
            del prob_im
            del occur_im
        preds = rearrange(preds, "c h w -> (h w) c")
        target = batch["mask"].to(device).flatten()
        # print(preds.get_device(), target.get_device())
        # print(preds.shape, target.shape)  # torch.Size([4002000, 16]) torch.Size([4002000])

        # accuracy.append(accuracy_metric(preds, target).item())
        jaccard += jaccard_metric(preds, target)  # adding the IoU per class
        # if confusion_matrix is None:
        #     confusion_matrix = confusion_matrix_metric(preds, target)
        # else:
        #     confusion_matrix += confusion_matrix_metric(preds, target)
        del x
        del batch
        del preds
        del target
        gc.collect()

    # print(len(accuracy), len(jaccard))  # 300 300

    ave_jac = jaccard / len(dataloader)  # average IoU per class
    print(ave_jac)
    ave_jac_specific_classes = torch.index_select(ave_jac, 0, indices)  # get classes
    print(ave_jac_specific_classes)
    print(torch.mean(ave_jac_specific_classes))  # calculate the final average
    # print(torch.std(ave_jac_specific_classes))

    print(ave_jac_specific_classes.cpu().numpy(), torch.mean(ave_jac_specific_classes).cpu().item())
    # print(np.mean(accuracy), np.std(accuracy))
    # print(confusion_matrix)
    return ave_jac_specific_classes.cpu().numpy(), torch.mean(ave_jac_specific_classes).cpu().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to yaml file")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to log directory containing checkpoint (usually lightning_logs/)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    predict_torch_metrics(args.config_file, args.log_dir, args.device)
