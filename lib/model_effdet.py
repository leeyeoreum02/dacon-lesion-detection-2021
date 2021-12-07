import numpy as np
import torch

from torch import optim
from effdet import get_efficientdet_config, EfficientDet
from effdet import DetBenchPredict, DetBenchTrain
from effdet.efficientdet import HeadNet

from pytorch_lightning import LightningModule
from ensemble_boxes import ensemble_boxes_wbf
from torchmetrics.detection.map import Metric, MAP


def create_model(
    mode,
    num_classes=4, 
    image_size=512,
    architecture='tf_efficientdet_d2',
):
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    if mode == 'train':
        return DetBenchTrain(net, config)
    elif mode == 'eval':
        return DetBenchPredict(net)
    else:
        raise Exception('parameter "mode" should be "train" or "eval".')


def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    wbf_boxes = []
    wbf_scores = []
    wbf_labels = []

    for prediction in predictions:
        boxes = [(prediction['boxes'] / image_size).tolist()]
        scores = [prediction['scores'].tolist()]
        labels = [prediction['classes'].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        wbf_boxes.append(boxes.tolist())
        wbf_scores.append(scores.tolist())
        wbf_labels.append(labels.tolist())

    return wbf_boxes, wbf_scores, wbf_labels


class EfficientDetModel(LightningModule):
    def __init__(
        self,
        mode,
        num_classes=4,
        image_size=512,
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        model_architecture='tf_efficientdet_b0',
        batch_size=32,
        skip_box_thr=0.44,
        metric: Metric = MAP,
    ):
        super().__init__()
        self.mode = mode
        self.image_size = image_size
        self.batch_size = batch_size
        self.skip_box_thr = skip_box_thr
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold

        self.model = create_model(
            mode=self.mode,
            num_classes=num_classes,
            image_size=image_size,
            architecture=model_architecture,
        )
        
        self.metric = metric()

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        losses = self.model(images, targets)

        self.log(
            'train_loss', losses['loss'], prog_bar=True, logger=True
        )
        self.log(
            'class_loss', losses['class_loss'], prog_bar=True, logger=True
        )
        self.log(
            'box_loss', losses['box_loss'], prog_bar=True, logger=True
        )

        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, targets)

        detections = outputs["detections"]

        targets = [{
            'boxes': boxes,
            'labels': labels.int(),
        } for boxes, labels in zip(targets['bbox'], targets['cls'])]
        
        detections = [{
            'boxes': detection[:, :4],
            'scores': detection[:, 4],
            'labels': detection[:, 5].int(),
        } for detection in detections]
        
        self.metric.update(preds=detections, target=targets)
    
    def validation_epoch_end(self, outputs) -> None:
        avg_map = self.metric.compute()
        
        self.log(
            'val_map50', avg_map['map_50'].detach(), 
            prog_bar=True, logger=True
        )
        
        logs = {"val_map50": avg_map['map_50'].detach()}
        
        self.metric.reset()
        
        return {"avg_val_iou": avg_map, "log": logs}
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets, image_ids = batch
        outputs = self.model(images)
        
        batch_boxes, batch_labels, batch_scores = \
            self._run_inference(outputs, targets['image_size'])
                    
        return image_ids, batch_boxes, batch_labels, batch_scores
            
    def _run_inference(self, detections, image_sizes):
        pred_boxes, pred_scores, pred_labels = \
            self.post_process_detections(detections)

        scaled_boxes = self._rescale_bboxes(
            predicted_bboxes=pred_boxes, image_sizes=image_sizes
        )

        return scaled_boxes, pred_labels, pred_scores

    def post_process_detections(self, detections):
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        
        for i in range(detections.shape[0]):
            predictions = self._postprocess_single_prediction_detections(detections[i])
            
            pred_boxes.append(predictions['boxes'])
            pred_scores.append(predictions['scores'])
            pred_labels.append(predictions['classes'])

        return pred_boxes, pred_scores, pred_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {'boxes': boxes, 'scores': scores[indexes], 'classes': classes[indexes]}

    def _rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_w, im_h = img_dims.cpu().detach().numpy()

            if len(bboxes) > 0:
                scaled_bboxes.append((
                    np.array(bboxes)
                    * [
                        im_w / self.image_size,
                        im_h / self.image_size,
                        im_w / self.image_size,
                        im_h / self.image_size,
                    ]).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes


if __name__ == '__main__':
    create_model()
