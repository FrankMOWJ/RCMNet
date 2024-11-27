import torch
from utils.dataloader import get_dataloader
from arch.dino_confidence import RCMNet
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
from datetime import datetime
from utils.helper import get_parser, seed_everything

best_mse = 9999.0
best_epoch = 0

def evaluate(model, dataloader, device, logger):
    model.eval()
    MAE, MSE = 0.0, 0.0
    with torch.no_grad():
        for (images, confidence_maps) in  dataloader:
            images, confidence_maps = images.to(device), confidence_maps.to(device)
            pred_confidence = model(images, confidence_maps).logits

            mae = F.l1_loss(pred_confidence, confidence_maps.unsqueeze(0))
            mse = F.mse_loss(pred_confidence, confidence_maps.unsqueeze(0))
            MSE += mse
            MAE += mae

    mean_MAE = MAE / len(dataloader)
    mean_MSE = MSE / len(dataloader)


    logger.info(f'MAE: {mean_MAE}, MSE: {mean_MSE}')
    return mean_MAE, mean_MSE


def train(model, train_loader, val_loader, optimizer, num_epochs, device, save_dir, logger):
    global best_mse, best_epoch
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for (images, confidence_maps) in train_loader:
            images, confidence_maps = images.to(device), confidence_maps.to(device)
            optimizer.zero_grad()
            outputs = model(images, confidence_maps)
            _, loss = outputs.logits, outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        mean_MAE, mean_MSE = evaluate(model, val_loader, device, logger)
        scheduler.step()

        if epoch % 1 == 0 and epoch != 0:
            checkpoint_path = f'{save_dir}/best.pt'
            if mean_MSE < best_mse:
                best_mse = mean_MSE
                best_epoch = epoch
                model.save_parameters(checkpoint_path)
        
    
    logger.info('Finished Training')

if __name__ == "__main__":
    args = get_parser()
    
    seed_everything(args.seed)
    confidence_head = args.confidence_head
    epoch = args.epoch
    batch_size = args.batch_size
    init_lr = args.init_lr
    DEVICE = args.device
    save_root = args.save_dir
    image_size = args.crop_size
    backbone_size = args.backbone_size
    data_root = args.data_root
    time_stamp = datetime.now().strftime('%Y%m%d-%H%M')
    name = f'{backbone_size}_{confidence_head}_{image_size}_lr{init_lr}_epoch{epoch}' if args.suffix is None else \
            f'{backbone_size}_{confidence_head}_{image_size}_lr{init_lr}_epoch{epoch}_{args.suffix}'
    
    save_dir =  rf'{save_root}/{name}/{time_stamp}'
    if not args.no_log:
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(rf'{save_dir}', exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f'{save_dir}/log.txt'),
                        logging.StreamHandler()
                    ] if not args.no_log else [
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)
    logger.info('************* Setting *************')
    logger.info(f'   Seed = {args.seed}')
    logger.info(f'   Image size = {args.crop_size}')
    logger.info(f'   Confidence normalization = {args.confidence_normalization}')
    logger.info(f'   Confidence head = {confidence_head}')
    logger.info(f'   Epoch = {epoch}')
    logger.info(f'   Batch size = {batch_size}')
    logger.info(f'   lr = {init_lr}')
    logger.info(f'************************************')
    # Instantiate Surgical-DINO
    model = RCMNet(backbone_size=backbone_size, r=4, lora_layer=None, image_shape=(image_size,image_size),  \
                        decode_type = 'linear4', confidence_head=args.confidence_head).to(DEVICE)
    
    # data
    train_loader, val_loader = get_dataloader(data_root, image_size, batch_size, args.confidence_normalization)

    
    if args.resume is not None:
        model.load_parameters(f'{args.resume}', DEVICE)
        print('evaluating...')
        evaluate(model, val_loader, DEVICE, logger)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, verbose=True)
    checkpoint_save_dir =  save_dir
    train(model, train_loader, val_loader, optimizer, epoch, DEVICE, checkpoint_save_dir, logger)

    model.save_parameters( f'{checkpoint_save_dir}/last.pt')
    logger.info(f'best val mse: {best_mse} at epoch {best_epoch}')
    logger.info(f'log and checkpoint are saved at {save_dir}')

