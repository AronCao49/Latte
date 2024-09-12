import os
import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from latte.data.utils.evaluate import Evaluator
from latte.data.utils.visualize import draw_points_image_labels
from latte.models.losses import entropy_loss, prob_2_entropy
from latte.data.collate import inverse_to_all, range_to_point
from latte.models.knn import KNN
from latte.data.utils.all_to_partial import pc2img_logit


def cross_modal_lifting(preds_2d, img_indices):
    img_feats = []
    for i in range(preds_2d.shape[0]):
        img_feats.append(preds_2d[i][img_indices[i][:, 0], img_indices[i][:, 1]])
    img_feats = torch.cat(img_feats, 0)

    return img_feats

def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             logger,
             pselab_dir=None,
             mix_match=False,
             entropy_fuse=False,
             save_pth=None,
             save_feat_pth=None):
    logger.info('Validation')

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_3d_all = Evaluator(class_names)
    evaluator_ensemble = Evaluator(class_names) if model_3d else None
    evaluator_ety = Evaluator(class_names) if entropy_fuse else None
    pselab_data_list = []

    # initialize KNN
    if cfg.VAL.use_knn:
        post_knn = KNN(cfg.MODEL_3D.NUM_CLASSES)
        post_knn = post_knn.cuda()
    else:
        post_knn = None
        
    feat_2d_stack = [[] for i in range(cfg.MODEL_2D.NUM_CLASSES)]
    feat_3d_stack = [[] for i in range(cfg.MODEL_3D.NUM_CLASSES)]

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            #! Bug test
            if iteration >= 20000:
                break
            
            data_time = time.time() - end
            # copy data from cpu to gpu
            data_batch['img'] = data_batch['img'].cuda()
            data_batch['seg_label'] = data_batch['seg_label'].cuda()
            # 3D input
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                if 'lidar' in data_batch.keys():
                    data_batch['lidar'] = data_batch['lidar'].cuda()
                else:
                    data_batch['x'][1] = data_batch['x'][1].cuda()
            elif 'RANGE' in cfg.DATASET_TARGET.TYPE:
                data_batch['proj_in'] = data_batch['proj_in'].float().cuda()
                data_batch['all_seg_label'] = data_batch['all_seg_label'].detach().numpy()
                if cfg.VAL.use_knn:
                    # * ONLY SUPPORT BATCH_SIZE = 1
                    assert cfg.VAL.BATCH_SIZE == 1
                    data_batch['proj_x'][0] = data_batch['proj_x'][0].cuda()
                    data_batch['proj_y'][0] = data_batch['proj_y'][0].cuda()
                    data_batch['proj_range'][0] = data_batch['proj_range'][0].cuda()
                    data_batch['unproj_range'][0] = data_batch['unproj_range'][0].cuda()
            else:
                raise NotImplementedError
            # For partial FOV
            data_batch['pc2img_idx'] = [idx.cuda() for idx in data_batch['pc2img_idx']]

            # predict
            if not mix_match:
                preds_2d = model_2d.inference(data_batch)
            else:
                preds_2d = model_2d.inference(data_batch['img'])
                preds_2d['seg_logit'] = cross_modal_lifting(preds_2d['seg_logit'], data_batch['img_indices'])
            preds_3d = model_3d(data_batch) if model_3d else None
            
            # mapping back to full label for SPVCNN
            if "SPVCNN" in cfg.MODEL_3D.TYPE:
                all_pred_logit_voxel_3d = inverse_to_all(preds_3d['seg_logit'], data_batch)
                pred_label_3d_all = all_pred_logit_voxel_3d.argmax(1).cpu().numpy()
                pred_logit_voxel_3d = pc2img_logit(all_pred_logit_voxel_3d, data_batch['pc2img_idx'])
                pred_label_voxel_3d = pred_logit_voxel_3d.argmax(1).cpu().numpy()
                if save_feat_pth is not None:
                    feats_3d = inverse_to_all(preds_3d['feats'], data_batch)
                    feats_3d = feats_3d.detach().cpu().numpy()
                    feats_2d = preds_2d['feats'].detach().cpu().numpy()
            elif cfg.MODEL_3D.TYPE == "SalsaNext":
                pred_logit_3d_all, pred_logit_voxel_3d, \
                pred_label_3d_all, pred_label_voxel_3d = range_to_point(preds_3d['seg_logit'], 
                                                                        data_batch,
                                                                        post_knn=post_knn,
                                                                        post=cfg.VAL.use_knn,
                                                                        output_prob=cfg.VAL.knn_prob)
                pred_label_3d_all = pred_label_3d_all.cpu().numpy()
                pred_label_voxel_3d = pred_label_voxel_3d.cpu().numpy()
            else:
                pred_logit_voxel_3d = preds_3d['seg_logit']
                pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            
            # get original point cloud from before voxelization
            # print(data_batch.keys())
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            
            # store feats separately based on pslabel
            # down sample 100 points
            rd_index = np.random.choice(pred_label_voxel_2d.shape[0], (300), replace=False)
            # rd_index = np.random.choice(pred_label_voxel_2d.shape[0], (pred_label_voxel_2d.shape[0]), replace=False)
            if save_feat_pth is not None:
                for label in np.unique(seg_label[0][rd_index]):
                    if label < 0:
                        continue
                    label_2d_mask = seg_label[0][rd_index] == label
                    label_3d_mask = seg_label[0][rd_index] == label
                    feat_2d_stack[label].extend(feats_2d[rd_index, :][label_2d_mask, :])
                    feat_3d_stack[label].extend(feats_3d[rd_index, :][label_3d_mask, :])
            
            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(pred_logit_voxel_3d, dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            val_2d_ety = prob_2_entropy(F.softmax(probs_2d, dim=1))
            val_3d_ety = prob_2_entropy(F.softmax(probs_3d, dim=1))
            val_metric_logger.update(val_2d_ety=torch.mean(val_2d_ety))
            val_metric_logger.update(val_3d_ety=torch.mean(val_3d_ety))
            if entropy_fuse:
                rv_ety_2d = 1 / (prob_2_entropy(probs_2d) + 1e-30)
                rv_ety_3d = 1 / (prob_2_entropy(probs_3d) + 1e-30)
                weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d)
                weight_3d = rv_ety_3d / (rv_ety_2d + rv_ety_3d)
                pred_label_ety_ensemble = (weight_2d * probs_2d + weight_3d * probs_3d).argmax(1).cpu().numpy()
                # print(pred_label_ensemble.shape, pred_label_ety_ensemble.shape)

            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                # For partial FOV
                curr_pc2img_idx = data_batch['pc2img_idx'][batch_ind].cpu().numpy()
                curr_seg_label = curr_seg_label[curr_pc2img_idx]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None
                pred_label_e_ensemble = pred_label_ety_ensemble[left_idx:right_idx] if entropy_fuse else None
                # print(pred_label_ensemble.shape, pred_label_ety_ensemble.shape)

                if save_pth is not None:
                    if not os.path.exists(save_pth):
                        os.makedirs(save_pth, exist_ok=True)
                    curr_image = data_batch['img'][batch_ind].detach().cpu().numpy()
                    curr_image = np.moveaxis(curr_image, 0, -1)
                    curr_seq, curr_id = data_batch['scene_name'][batch_ind], data_batch['scan_ID'][batch_ind]
                    draw_points_image_labels(
                        curr_image, data_batch['img_indices'][batch_ind], pred_label_2d,
                        color_palette_type='SemanticKITTI', 
                        save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "2D")))
                    draw_points_image_labels(
                        curr_image, data_batch['img_indices'][batch_ind], pred_label_3d,
                        color_palette_type='SemanticKITTI', 
                        save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "3D")))   
                    draw_points_image_labels(
                        curr_image, data_batch['img_indices'][batch_ind], pred_label_ensemble,
                        color_palette_type='SemanticKITTI', 
                        save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "xM")))
                    draw_points_image_labels(
                        curr_image, data_batch['img_indices'][batch_ind], curr_seg_label,
                        color_palette_type='SemanticKITTI', 
                        save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "GT")))  
                
                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)
                    if entropy_fuse:
                        evaluator_ety.update(pred_label_e_ensemble, curr_seg_label)

                left_idx = right_idx
            
            # update evaluator for all points
            if evaluator_3d_all:
                evaluator_3d_all.update(pred_label_3d_all, data_batch['seg_label'].cpu().numpy())

            data_batch['seg_label'] = pc2img_logit(data_batch['seg_label'], data_batch['pc2img_idx'])
            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label'])
            seg_loss_3d = F.cross_entropy(pred_logit_voxel_3d, data_batch['seg_label']) if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
            if evaluator_3d_all:
                eval_list.extend([('3D all', evaluator_3d_all)])
            if entropy_fuse:
                eval_list.extend([('entropy fuse', evaluator_ety)])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        # compute feats summary
        if save_feat_pth is not None:
        # 2D feats
            ext_2d_mu = []
            ext_2d_var = []
            for feat in feat_2d_stack:
                feat = np.stack(feat)
                feat = feat / np.linalg.norm(feat, axis=1).reshape(-1,1)
                ext_2d_mu.append(np.mean(feat, axis=0))
                ext_2d_var.append(np.var(feat, axis=0))
                # ext_2d_cov.append(covariance(np.stack(feat)))
            # 3D feats
            ext_3d_mu = []
            ext_3d_var = []
            for feat in feat_3d_stack:
                feat = np.stack(feat)
                feat = feat / np.linalg.norm(feat, axis=1).reshape(-1,1)
                ext_3d_mu.append(np.mean(feat, axis=0))
                ext_3d_var.append(np.var(feat, axis=0))
                # ext_3d_cov.append(covariance(np.stack(feat)))
            # save to feat_pth
            feats_dict = {
                "mu_2d": ext_2d_mu, "var_2d": ext_2d_var,
                "mu_3d": ext_3d_mu, "var_3d": ext_3d_var,
            }
            np.save(save_feat_pth, feats_dict)  

    eval_dict = {key: evaluator for key, evaluator in eval_list}
    
    return eval_dict
