import numpy as np
import torch
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, wb_mask, CLASSES
import wandb

def val(args, model, dataloader, final_test):
    # init wandb artifacts
    # save validation predictions, create a new version of the artifact for each epoch
    val_res_at = wandb.Artifact("val_pred_" + wandb.run.id, "val_epoch_preds")
    # store all final results in a single artifact across experiments and
    # model variants to easily compare predictions
    final_model_res_at = wandb.Artifact("bisenet_pred", "model_preds")
    main_columns = ["prediction", "ground_truth"]
    # we'll track the IOU for each class
    main_columns.extend(["iou_" + s for s in CLASSES])
    # create tables
    val_table = wandb.Table(columns=main_columns)
    model_res_table = wandb.Table(columns=main_columns)
    
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            current_hist = fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            hist += current_hist

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
            
            # add row to the wandb table
            row = [wb_mask(data, pred_mask=predict), wb_mask(data, true_mask=label)]
            row.extend(per_class_iu(current_hist))
            val_table.add_data(*row)
            if final_test == True:
                model_res_table.add_data(*row)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print(f'precision per pixel for test: {precision:.3f}')
        print(f'mIoU for validation: {miou:.3f}')
        print(f'mIoU per class: {miou_list}')
        
        # upload wandb table
        val_res_at.add(val_table, "val_epoch_results")
        wandb.run.log_artifact(val_res_at)
        if final_test == True:
            final_model_res_at.add(model_res_table, "model_results")
            wandb.run.log_artifact(final_model_res_at)

        return precision, miou
