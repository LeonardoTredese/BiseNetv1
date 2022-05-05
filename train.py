import os
import numpy as np
import torch
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils import poly_lr_scheduler
from loss import DiceLoss
from tqdm import tqdm
import wandb


def segmentation_train(args, model, optimizer, dataloader_train, dataloader_val, device):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    scaler = amp.GradScaler()
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.init_epoch, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            with amp.autocast():
                data, label = data.to(device), label.to(device)
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        wandb.log({"loss": loss_train_mean})
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        is_right_iteration = lambda i, step: (i % step) == (step - 1)
        if is_right_iteration(epoch, args.checkpoint_step) and args.model_file_name is not None:
            model_path = os.path.join(args.saved_models_path, args.model_file_name)
            torch.save(model.module.state_dict(), model_path) 

        if is_right_iteration(epoch, args.validation_step):
            precision, miou = val(args, model, dataloader_val, False)
            if miou > max_miou and args.model_file_name is not None:
                max_miou = miou
                best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
                torch.save(model.module.state_dict(), best_model_path) 
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)

