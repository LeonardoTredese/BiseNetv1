python main.py  --num_epochs 50 \
        --segmentation_lr 2.5e-3 \
        --discriminator_lr 1e-3 \
        --data ../data/ \
        --num_classes 19 \
        --cuda 0 \
	--validation_step 100 \
	--num_workers 4 \
        --batch_size 4 \
        --adapt_domain \
        --source_dataset GTA5 \
        --target_dataset Cityscapes \
        --validation_dataset Cityscapes \
        --context_path resnet18 \
        --segmentation_loss crossentropy \
        --adversarial_loss bce \
        --saved_models_path ./checkpoints_adv_18_sgd \
	--model_file_name bisenet.torch \
        --segmentation_optimizer sgd \
        --discriminator_optimizer adam \
        --adversarial_lambda .001 \
        --depthwise_discriminator
