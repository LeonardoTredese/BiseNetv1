python main.py  --num_epochs 50 \
        --segmentation_lr 2.5e-2 \
        --data ../data/ \
        --num_classes 19 \
        --cuda 0 \
	--validation_step 100 \
	--num_workers 4 \
        --batch_size 8 \
        --no-adapt_domain \
        --source_dataset GTA5 \
        --validation_dataset Cityscapes \
        --context_path resnet18 \
        --segmentation_loss crossentropy \
        --saved_models_path ./checkpoints_18_sgd \
	--model_file_name bisenet.torch \
        --segmentation_optimizer sgd \
