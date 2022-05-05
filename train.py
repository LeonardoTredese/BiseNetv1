import os
import numpy as np
import torch
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils import poly_lr_scheduler
from loss import DiceLoss
from tqdm import tqdm
import wandb


def adversarial_train(args, model, discriminator, model_optimizer, discriminator_optimizer,\
                        source_loader_train, target_loader_train,target_loader_val, device):
    writer = SummaryWriter(comment=f'adversarial, {args.optimizer}, {args.context_path}')
    scaler = amp.GradScaler()
    max_miou, step = 0, 0
    is_source, is_target = 1.0, 0.0 
      
    if args.loss == 'dice':
        model_loss = DiceLoss()
    elif args.loss == 'crossentropy':
        model_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    discriminator_loss = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(args.init_epoch, args.num_epochs):
        # Imposing same learning rate for discriminator and model 
        model_lr = poly_lr_scheduler(model_optimizer,\
                                        args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        discriminator_lr = poly_lr_scheduler(discriminator_optimizer, \
                                        args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        tq = tqdm(total=len(target_loader_train) * args.batch_size)
        tq.set_description('adv, epoch %d, lr %f' % (epoch, model_lr))
        seg_loss_record, adv_loss_record, disc_loss_record = [], [], []
        model.train()
        for (s_image, s_label), (t_image, t_label) in zip(source_loader_train, target_loader_train):
            model_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            with amp.autocast():
                s_image, s_label = s_image.to(device), s_label.to(device) 
                t_image, t_label = t_image.to(device), t_label.to(device)
                 
            # train on source domain
            with amp.autocast():
                seg_source, output_sup1, output_sup2 = model(t_image)
                loss1 = loss_func(seg_source, s_label)
                loss2 = loss_func(output_sup1, s_label)
                loss3 = loss_func(output_sup2, s_label)
                seg_loss = loss1 + loss2 + loss3
            seg_loss_record.append(seg_loss.item())
            scaler.scale(seg_loss).backward()
            scaler.step(model_optimizer)
            scaler.update()
           
            # confuse discriminator
            with amp.autocast():
                seg_target, _, _ = model(t_image)
                with torch.no_grad():
                    d_out = discriminator(seg_target)
                confused_label = is_source * tensor.ones(d_out.data.size(), device=device)
                adv_loss = args.adversaria_lambda * discriminator_loss(d_out, confused_label)
            adv_loss_record.append(adv_loss.item())
            scaler.scale(adv_loss).backward()
            scaler.step(model_optimizer)
            scaler.update()

            # train discriminator
            with amp.autocast():
                s_out, t_out  = discriminator(seg_source.detach()), discriminator(seg_target.detach())
                s_disc_label = is_source * tensor.ones(s_out.data.size(), device=device)
                t_disc_label = is_target * tensor.ones(t_out.data.sise(), device=device)
                s_disc_loss = discriminator_loss(s_out, s_disc_label)
                t_disc_loss = discriminator_loss(t_out, t_disc_label)
                disc_loss = (s_disc_loss + t_disc_loss) / 2
            disc_loss_record.append(disc_loss.item()) 
            scaler.scale(disc_loss).backward()
            scaler.step(discriminator_optimizer)
            scaler.update()
                
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        seg_loss_train_mean = np.mean(seg_loss_record)
        adv_loss_train_mean = np.mean(adv_loss_record)
        disc_loss_train_mean = np.mean(disc_loss_record)
        wandb.log({"segmentation loss": seg_loss_train_mean})
        wandb.log({"adversarial loss": adv_loss_train_mean})
        wandb.log({"discriminator loss": disc_loss_train_mean})
        writer.add_scalar('epoch/seg_loss_train_mean', float(seg_loss_train_mean), epoch)
        writer.add_scalar('epoch/adv_loss_train_mean', float(adv_loss_train_mean), epoch)
        writer.add_scalar('epoch/disc_loss_train_mean', float(disc_loss_train_mean), epoch)
        print(f'segmentation loss: {seg_loss_train_mean}')
        print(f'adversarial loss: {adv_loss_train_mean}' )
        print(f'discriminator loss: {disc_loss_train_mean}') 
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

