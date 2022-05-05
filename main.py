import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
from dataset import dataset
import utils
import wandb
from train import segmentation_train
from validate import val

def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--init_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--model_file_name', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--saved_models_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')

    args = parser.parse_args()

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.nn.DataParallel(model, output_device = device)
    
    # Create datasets instance
    dataset_path = os.path.join(args.data, args.dataset)
    new_size = (args.crop_height, args.crop_width)
    if args.dataset == 'Cityscapes':
        dataset_train = dataset.Cityscapes(dataset_path, new_size, 'train')
        dataset_val = dataset.Cityscapes(dataset_path, new_size, 'val')
    elif args.dataset == 'GTA5':
        dataset_train = dataset.Gta5(dataset_path, new_size, 'train', )
        # Validation just for trainin pourposes
        dataset_val = [dataset_train[0]]
    else:
        raise Exception('Please choose either Cityscapes or GTA5 as datasets')    

    # Define your dataloaders:
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

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
    segmentation_train(args, model, optimizer, dataloader_train, dataloader_val, device)
    # final test
    val(args, model, dataloader_val, True)
    
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
