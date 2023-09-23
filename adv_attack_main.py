from pathlib import Path
import argparse
import torch
import torchattacks
from torch.utils.data import DataLoader
import numpy as np
import random
import datasets
import util.misc as utils
from util.argparse import get_args_parser
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import cv2
from datasets.torchvision_datasets import CocoDetection
from torchattacks import attack
@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load the model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model
    checkpoint = torch.load(args.resume, map_location='cpu')

    #load the dataset
    dataset_train:CocoDetection = build_dataset(image_set='train', args=args)
    dataset_val:CocoDetection = build_dataset(image_set='val', args=args)
    
    # load the pretrained value
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    
    #PGD attack
    instance_pgd = torchattacks.PGD(model_without_ddp,eps=8/255,alpha=1/255,steps=10,random_start=True)
    instance_pgd.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    print(instance_pgd) 
    picked_image, picked_label = dataset_train[3]
    picked_image = torch.cat([picked_image])
    picked_labels = torch.cat([picked_label["labels"]])
    adv_images = instance_pgd(picked_image,picked_labels)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)