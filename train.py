import os
import numpy as np
import torch
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils import poly_lr_scheduler, FDA_source_to_target, self_entropy
from validate import val
from loss import DiceLoss
from tqdm import tqdm
import wandb

def get_loss(loss_name):
    if loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'crossentropy':
        return torch.nn.CrossEntropyLoss(ignore_index=255)
    elif loss_name == 'leastsquares':
        return torch.nn.MSELoss()
    elif loss_name == 'bce':
        return torch.nn.BCEWithLogitsLoss() 

def adversarial_train(args, model, discriminator, model_optimizer, discriminator_optimizer,\
                        source_loader_train, target_loader_train, target_loader_val, device):
    writer = SummaryWriter(comment=f'Segmentation, {args.segmentation_optimizer}, {args.context_path}, Discriminator {args.discriminator_optimizer}')
    scaler = amp.GradScaler()
    max_miou, step = 0, 0
    is_source, is_target = .0, 1.0 
      
    model_loss = get_loss(args.segmentation_loss)
    discriminator_loss = get_loss(args.adversarial_loss)
    
    for epoch in range(args.init_epoch, args.num_epochs):
        # Imposing same learning rate for discriminator and model 
        model_lr = poly_lr_scheduler(model_optimizer,\
                                        args.segmentation_lr, iter=epoch, max_iter=args.num_epochs)
        discriminator_lr = poly_lr_scheduler(discriminator_optimizer, \
                                        args.discriminator_lr, iter=epoch, max_iter=args.num_epochs)
        
        tq = tqdm(total=len(target_loader_train.dataset))
        tq.set_description(f'adversarial, epoch {epoch}, seg_lr {model_lr:.4e}, discriminator_lr {discriminator_lr:.4e} ')
        
        seg_loss_record, adv_loss_record, disc_loss_record = [], [], []
        
        model.train()
        discriminator.train()
        for (s_image, s_label), (t_image, t_label) in zip(source_loader_train, target_loader_train):
            model_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            
            with amp.autocast():
                s_image, s_label = s_image.to(device), s_label.to(device) 
                t_image, t_label = t_image.to(device), t_label.to(device)
                t_image = FDA_source_to_target(t_image, s_image, L = 0.003) 
            # train on source domain
            with amp.autocast():
                seg_source, output_sup1, output_sup2 = model(s_image)
                loss1 = model_loss(seg_source, s_label)
                loss2 = model_loss(output_sup1, s_label)
                loss3 = model_loss(output_sup2, s_label)
                seg_loss = loss1 + loss2 + loss3
            seg_loss_record.append(seg_loss.item())
            scaler.scale(seg_loss).backward()
           
            # confuse discriminator
            if device == torch.device('cpu'):
                discriminator.requires_grad(False)
            else:
                discriminator.module.requires_grad(False)
            
            with amp.autocast():
                seg_target, _, _ = model(t_image)
                d_out = discriminator(seg_target)
                confused_label = is_source * torch.ones(d_out.data.size(), device=device)
                adv_loss = args.adversarial_lambda * discriminator_loss(d_out, confused_label)
            adv_loss_record.append(adv_loss.item())
            scaler.scale(adv_loss).backward()
            scaler.step(model_optimizer)

            # train discriminator
            if device == torch.device('cpu'):
                discriminator.requires_grad(True)
            else:
                discriminator.module.requires_grad(True)
            with amp.autocast():
                s_out = discriminator(seg_source.detach())
                s_disc_label = is_source * torch.ones(s_out.data.size(), device=device)
                s_disc_loss = discriminator_loss(s_out, s_disc_label) / 2
                t_out = discriminator(seg_target.detach())
                t_disc_label = is_target * torch.ones(t_out.data.size(), device=device)
                t_disc_loss = discriminator_loss(t_out, t_disc_label) / 2
            disc_loss = s_disc_loss.item() + t_disc_loss.item()
            disc_loss_record.append(disc_loss) 
            
            scaler.scale(s_disc_loss).backward()
            scaler.scale(t_disc_loss).backward()
            scaler.step(discriminator_optimizer)
            scaler.update()
            
            step += 1
            writer.add_scalar('seg_loss_step', seg_loss, step)
            tq.update(s_image.shape[0])
            tq.set_postfix(seg_loss=f'{seg_loss:.4e}', adv_loss=f'{adv_loss:.4e}', disc_loss=f'{disc_loss:.4e}')
        tq.close()
        seg_loss_train_mean = np.mean(seg_loss_record)
        adv_loss_train_mean = np.mean(adv_loss_record)
        disc_loss_train_mean = np.mean(disc_loss_record)
        wandb.log({"segmentation loss":  seg_loss_train_mean, \
                   "adversarial loss":   adv_loss_train_mean, \
                   "discriminator loss": disc_loss_train_mean})
        writer.add_scalar('epoch/seg_loss_train_mean', float(seg_loss_train_mean), epoch)
        writer.add_scalar('epoch/adv_loss_train_mean', float(adv_loss_train_mean), epoch)
        writer.add_scalar('epoch/disc_loss_train_mean', float(disc_loss_train_mean), epoch)
        is_right_iteration = lambda i, step: (i % step) == (step - 1)
        if is_right_iteration(epoch, args.checkpoint_step) and args.model_file_name is not None:
            model_path = os.path.join(args.saved_models_path, args.model_file_name)
            torch.save(model.module.state_dict(), model_path) 

        if is_right_iteration(epoch, args.validation_step):
            precision, miou = val(args, model, target_loader_val, False)
            if miou > max_miou and args.model_file_name is not None:
                max_miou = miou
                best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
                torch.save(model.module.state_dict(), best_model_path) 
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)
                

def FDA_train(args, model, model_optimizer, source_loader_train, \
              target_loader_train, target_loader_val, invert_target_source, beta, device):
    writer = SummaryWriter(comment=f'FDA + Segmentation, {args.segmentation_optimizer}, {args.context_path}')
    scaler = amp.GradScaler()
    max_miou, step = 0, 0
    is_source, is_target = .0, 1.0 
      
    model_loss = get_loss(args.segmentation_loss)
    
    for epoch in range(args.init_epoch, args.num_epochs):
        # Imposing same learning rate for discriminator and model 
        model_lr = poly_lr_scheduler(model_optimizer,\
                                        args.segmentation_lr, iter=epoch, max_iter=args.num_epochs)
        
        tq = tqdm(total=len(target_loader_train.dataset))
        tq.set_description(f'FDA epoch {epoch}, seg_lr {model_lr:.4e}')
        
        src_loss_record, trg_loss_record = [], []
        
        model.train()
        for (s_image, s_label), (t_image, t_label) in zip(source_loader_train, target_loader_train):
            model_optimizer.zero_grad()
            
            with amp.autocast():
                s_image, s_label = s_image.to(device), s_label.to(device) 
                t_image, t_label = t_image.to(device), t_label.to(device)
                if invert_target_source:
                    t_image = FDA_source_to_target(t_image, s_image, L = beta) 
                else:
                    s_image = FDA_source_to_target(s_image, t_image, L = beta) 
            # train on source domain
            with amp.autocast():
                seg_source, output_sup1, output_sup2 = model(s_image)
                s_loss1 = model_loss(seg_source, s_label)
                s_loss2 = model_loss(output_sup1, s_label)
                s_loss3 = model_loss(output_sup2, s_label)
                src_loss = s_loss1 + s_loss2 + s_loss3
            src_loss_record.append(src_loss.item())

            # train on target domain
            with amp.autocast():
                seg_source, output_sup1, output_sup2 = model(t_image)
                t_loss1 = self_entropy(seg_source)
                t_loss2 = self_entropy(output_sup1)
                t_loss3 = self_entropy(output_sup2)
                trg_loss = (t_loss1 + t_loss2 + t_loss3) / 3
            trg_loss_record.append(trg_loss.item())
            scaler.scale(src_loss + 0.005 * trg_loss).backward()
            scaler.step(model_optimizer)
            scaler.update()
            
            step += 1
            writer.add_scalar('src_loss_step', src_loss, step)
            # writer.add_scalar('trg_loss_step', trg_loss, step)
            tq.update(s_image.shape[0])
            tq.set_postfix(src_loss=f'{src_loss:.4e}')#,trg_loss=f'{trg_loss:.4e}')
        tq.close()
        seg_loss_train_mean = np.mean(src_loss_record)
        wandb.log({"segmentation loss":  seg_loss_train_mean})
        writer.add_scalar('epoch/seg_loss_train_mean', float(seg_loss_train_mean), epoch)
        is_right_iteration = lambda i, step: (i % step) == (step - 1)
        if is_right_iteration(epoch, args.checkpoint_step) and args.model_file_name is not None:
            model_path = os.path.join(args.saved_models_path, args.model_file_name)
            torch.save(model.module.state_dict(), model_path) 

        if is_right_iteration(epoch, args.validation_step):
            precision, miou = val(args, model, target_loader_val, False)
            if miou > max_miou and args.model_file_name is not None:
                max_miou = miou
                best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
                torch.save(model.module.state_dict(), best_model_path) 
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)

def FDA_train_ranking(args, model, model_optimizer, source_loader_train, \
              target_loader_train, target_loader_val, invert_target_source, beta, device):
    writer = SummaryWriter(comment=f'FDA + Segmentation, {args.segmentation_optimizer}, {args.context_path}')
    scaler = amp.GradScaler()
    max_miou, step = 0, 0
    is_source, is_target = .0, 1.0 
      
    model_loss = get_loss(args.segmentation_loss)
    
    for epoch in range(args.init_epoch, args.num_epochs):
        # Imposing same learning rate for discriminator and model 
        model_lr = poly_lr_scheduler(model_optimizer,\
                                        args.segmentation_lr, iter=epoch, max_iter=args.num_epochs)
        
        tq = tqdm(total=len(target_loader_train.dataset))
        tq.set_description(f'FDA epoch {epoch}, seg_lr {model_lr:.4e}')
        
        src_loss_record, trg_loss_record = [], []
        
        model.train()
        for (s_image, s_label), (t_image, t_label) in zip(source_loader_train, target_loader_train):
            model_optimizer.zero_grad()
            
            # train on source domain
            with amp.autocast():
                seg_source, output_sup1, output_sup2 = model(s_image)
                s_loss1 = model_loss(seg_source, s_label)
                s_loss2 = model_loss(output_sup1, s_label)
                s_loss3 = model_loss(output_sup2, s_label)
                src_loss = s_loss1 + s_loss2 + s_loss3
            src_loss_record.append(src_loss.item())

            # train on target domain
            with amp.autocast():
                seg_source, output_sup1, output_sup2 = model(t_image)
                t_loss1 = self_entropy(seg_source)
                t_loss2 = self_entropy(output_sup1)
                t_loss3 = self_entropy(output_sup2)
                trg_loss = (t_loss1 + t_loss2 + t_loss3) / 3
            trg_loss_record.append(trg_loss.item())
            scaler.scale(src_loss + 0.005 * trg_loss).backward()
            scaler.step(model_optimizer)
            scaler.update()
            
            step += 1
            writer.add_scalar('src_loss_step', src_loss, step)
            # writer.add_scalar('trg_loss_step', trg_loss, step)
            tq.update(s_image.shape[0])
            tq.set_postfix(src_loss=f'{src_loss:.4e}')#,trg_loss=f'{trg_loss:.4e}')
        tq.close()
        seg_loss_train_mean = np.mean(src_loss_record)
        wandb.log({"segmentation loss":  seg_loss_train_mean})
        writer.add_scalar('epoch/seg_loss_train_mean', float(seg_loss_train_mean), epoch)
        is_right_iteration = lambda i, step: (i % step) == (step - 1)
        if is_right_iteration(epoch, args.checkpoint_step) and args.model_file_name is not None:
            model_path = os.path.join(args.saved_models_path, args.model_file_name)
            torch.save(model.module.state_dict(), model_path) 

        if is_right_iteration(epoch, args.validation_step):
            precision, miou = val(args, model, target_loader_val, False)
            if miou > max_miou and args.model_file_name is not None:
                max_miou = miou
                best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
                torch.save(model.module.state_dict(), best_model_path) 
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)          

def segmentation_train(args, model, optimizer, dataloader_train, dataloader_val, device):
    writer = SummaryWriter(comment=f'Segmentation, {args.segmentation_optimizer}, {args.context_path}')
    scaler = amp.GradScaler()
    
    loss_func = get_loss(args.segmentation_loss)
    
    max_miou = 0
    step = 0
    for epoch in range(args.init_epoch, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.segmentation_lr, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train.dataset))
        tq.set_description(f'segmentation, epoch {epoch}, seg_lr {lr:.4e}')
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
            tq.set_postfix(seg_loss=f'{loss:.4e}')
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        wandb.log({"segmentation loss": loss_train_mean})
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
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

