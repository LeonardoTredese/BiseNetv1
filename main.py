from functools import partial
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
from model.discriminator import DepthwiseDiscriminator, FCDiscriminator 
from dataset import dataset
import utils
import wandb
from train import segmentation_train, adversarial_train, FDA_train, FDA_train_ranking
from validate import val

def get_dataset(dataset_name, crop_size, augment_data , task, base_path):
    path = os.path.join(base_path, dataset_name)
    if dataset_name == 'GTA5':
        return dataset.Gta5(path, crop_size, augment_data, task)
    elif dataset_name == 'Cityscapes':
        return dataset.Cityscapes(path, crop_size, augment_data, task)

def get_optimizer(optimizer, model, learning_rate):
    params = model.parameters()
    if optimizer == 'rmsprop':
        return torch.optim.RMSprop(params, learning_rate)
    elif optimizer == 'sgd':
        return torch.optim.SGD(params, learning_rate, momentum=.9, weight_decay=1e-4)
    elif optimizer == 'adam':
        return torch.optim.Adam(params, learning_rate, betas=(.9, .99))

def main():
    # basic parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_type', type=str, help='Available training types: "SEG" normal segmentation, "ADV_DA" adversarial domain adaptation, "FDA" fourier domain adaptation, "FDA_VAL" fourier domain adaptation during validation')
    parser.add_argument('--fda_inverted', action='store_true', help='Perform target to source fourier doman adaptation')
    parser.set_defaults(fda_inverted=False)
    parser.add_argument('--fda_beta', type=float, default=3e-3, help='Beta parameter that regulates how many frequencies to swap between source and target')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--init_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--source_dataset', type=str, default="GTA5", help='Source domain dataset')
    parser.add_argument('--target_dataset', type=str, default="Cityscapes", help='Target domain dataset')
    parser.add_argument('--validation_dataset', type=str, default="Cityscapes", help='Dataset to perform validation on')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",help='The context path model you are using: either resnet18, resnet101.')
    parser.add_argument('--segmentation_lr', type=float, default=2.5e-4, help='learning rate used to train the segmentation network')
    parser.add_argument('--discriminator_lr', type=float, default=1e-4, help='learning rate used to train the adversarial network')
    parser.add_argument('--data', type=str, default='', help='base path of training data')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--model_file_name', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--saved_models_path', type=str, default=None, help='path to save model')
    parser.add_argument('--segmentation_optimizer', type=str, default='rmsprop', help='optimizer for segmentation network, support rmsprop, sgd, adam')
    parser.add_argument('--discriminator_optimizer', type=str, default='rmsprop', help='optimizer for adversarial network, support rmsprop, sgd, adam')
    parser.add_argument('--segmentation_loss', type=str, default='crossentropy', help='loss function for segmentation, dice or crossentropy')
    parser.add_argument('--adversarial_loss', type=str, default='crossentropy', help='loss function for segmentation, dice or crossentropy')
    parser.add_argument('--adversarial_lambda', type=float, default=.001, help='Multiplication constant for adversarial loss')
    parser.add_argument('--depthwise_discriminator', action='store_true', help='Perform depthwise convolution in discriminator, disable with --no-depthwise_discriminator')
    parser.add_argument('--no-depthwise_discriminator', dest='depthwise_discriminator', action='store_false')
    parser.set_defaults(depthwise_discriminator=False)
    parser.add_argument('--augment_data', action='store_true', help='Perform data augmentation, disable with --no-augment_data')
    parser.add_argument('--no-augment_data', dest='augment_data', action='store_false')
    parser.set_defaults(augment_data=False)
    parser.add_argument('--ranking', type=str, default=None, help='Ranking for choosing target image in FDA')

    args = parser.parse_args()

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if args.train_type == 'ADV_DA':
        discriminator = DepthwiseDiscriminator(args.num_classes) if args.depthwise_discriminator else FCDiscriminator(args.num_classes)

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
        model = torch.nn.DataParallel(model, output_device = device)
        if args.train_type == 'ADV_DA':
            discriminator = torch.nn.DataParallel(discriminator, output_device = device)
    else:
        device = torch.device('cpu')
    
    # Create datasets instance
    crop_size = (args.crop_height, args.crop_width)
    if args.train_type in ['ADV_DA', 'FDA']:
        source_dataset = dataset.FDADataset(args.source_dataset, args.target_dataset, args.data, crop_size, args.augment_data, 'train', args.fda_beta, args.ranking)
        target_dataset = get_dataset(args.target_dataset, crop_size, args.augment_data, 'train', args.data)
    else:
        source_dataset = get_dataset(args.source_dataset, crop_size, args.augment_data, 'train',args.data)
    if args.train_type == "FDA_VAL":
        validation_dataset = dataset.FDADataset(args.validation_dataset, args.source_dataset, args.data, crop_size, args.augment_data, 'val', args.fda_beta, args.ranking)
    else:
        validation_dataset = get_dataset(args.validation_dataset, crop_size, False, 'val', args.data)

    # Define your dataloaders:
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    validation_loader= DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    if args.train_type in ['ADV_DA', 'FDA']:
        target_loader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers
        )

    # build optimizer
    model_optimizer = get_optimizer(args.segmentation_optimizer, model, args.segmentation_lr)
    if args.train_type == 'ADV_DA':
        discriminator_optimizer = get_optimizer(args.discriminator_optimizer, discriminator, args.segmentation_lr)

    # load pretrained model if exists
    if args.saved_models_path is not None:
        os.makedirs(args.saved_models_path, exist_ok=True)
    if args.model_file_name is not None:
        assert args.saved_models_path is not None, 'If you specify model file name, you must specify a directory'
        model_path = os.path.join(args.saved_models_path, args.model_file_name)
        if os.path.exists(model_path):
            print(f'Loading state from file: {model_path}')            
            model.module.load_state_dict(torch.load(model_path))
        else:
            print(f'Could not find model {model_path}, starting from zero')
    else:
        print('[WARNING] Model name was not specified. No data will be stored after training')

    # wandb logging init
    wandb.login()
    run = wandb.init(project=utils.WANDB_PROJECT, entity=utils.WANDB_ENTITY, job_type="train", config=args)
    wandb.watch(model, log_freq=args.batch_size)
    
    # train
    if args.train_type == 'ADV_DA':
        adversarial_train(args, model, discriminator, model_optimizer, discriminator_optimizer, \
                          source_loader, target_loader, validation_loader, device)
    elif args.train_type == 'FDA':
        # FDA_train(args, model, model_optimizer, source_loader, target_loader, validation_loader, args.fda_inverted, args.fda_beta, device)
        FDA_train_ranking(args, model, model_optimizer, source_loader, target_loader, validation_loader, args.fda_inverted, args.fda_beta, device)
    else:
        segmentation_train(args, model, model_optimizer, source_loader, validation_loader, device)
    
    # final test
    val(args, model, validation_loader, True)
    
    # save model in wandb and close connection
    if args.saved_models_path is not None and args.model_file_name is not None:
        best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
        if os.path.exists(best_model_path ):
            model_name = args.model_file_name
            saved_model = wandb.Artifact(model_name, type="model")
            saved_model.add_file(best_model_path, name=model_name)
            print("Saving data to WandB...")
            run.log_artifact(saved_model)
    run.finish()
    print("Completed run")


if __name__ == '__main__':
    main()
