import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
from lib.utils.utils import model_info
from plateNet import myNet_ocr
from  alphabets import plateName,plate_chr
from LPRNet import build_lprnet

from tensorboardX import SummaryWriter

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    
    parser.add_argument('--cfg', help='/mnt/jx/car_plate_rec/lib/config/crnn_config.yaml', required=True, type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height') 
    parser.add_argument('--img_w',type=int,default=168,help='width')
    args = parser.parse_args()
   
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.HEIGHT=args.img_h
    config.WIDTH = args.img_w
    return config


def main():
    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # model configuration
    cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]  # big model
    model = myNet_ocr(num_classes=len(plate_chr), cfg=cfg)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch - 1
    )

    # Load checkpoint if resume or finetune
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        checkpoint = torch.load(config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT, map_location='cpu')
        model.load_state_dict(checkpoint)
    elif config.TRAIN.RESUME.IS_RESUME:
        checkpoint = torch.load(config.TRAIN.RESUME.FILE, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']

    model_info(model)
    train_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model,
                       criterion, optimizer, device, epoch, writer_dict, output_dict)

        lr_scheduler.step()

        # Evaluate every 50 epochs
        if epoch % 10 == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            acc = function.validate(config, val_loader, val_dataset, converter,
                                    model, criterion, device, epoch, writer_dict, output_dict)

            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                print(f"New best accuracy: {best_acc:.4f} at epoch {epoch}")

                # save best checkpoint
                torch.save(
                    {
                        "cfg": cfg,
                        "state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_acc": best_acc,
                    }, os.path.join(output_dict['chs_dir'], f"best_checkpoint_epoch_{epoch}_acc_{best_acc:.4f}.pth")
                )

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

# python train2.py --cfg /data_249/data3/plate_rec/work/car_plate_rec/lib/config/360CC_config.yaml > output.txt