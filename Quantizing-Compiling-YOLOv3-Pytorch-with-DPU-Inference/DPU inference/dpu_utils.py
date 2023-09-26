import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exp(x):
    return np.exp(x)

class YOLOPost:
    def __init__(self, anchors, num_classes, img_size):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

    def forward(self, input, targets=None):
        bs = input.shape[0]
        in_h = input.shape[2]
        in_w = input.shape[3]
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        # scaled_anchors = np.array([[a_w / stride_w, a_h / stride_h] for a_w, a_h in self.anchors])

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]


        # prediction = input.view(bs,  self.num_anchors,
        #                         self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        prediction = input.reshape(bs, self.num_anchors, self.bbox_attrs, in_h, in_w).transpose(0, 1, 3, 4, 2)
        print("Prediction shape: ", prediction.shape)

        # Get outputs
        x = sigmoid(prediction[..., 0])          # Center x
        y = sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                   # Width
        h = prediction[..., 3]                   # Height
        conf = sigmoid(prediction[..., 4])       # Conf
        pred_cls = sigmoid(prediction[..., 5:])  # Cls pred.


        print("forward being called")

        # Calculate offsets for each grid
        # grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
        #     bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        # grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
        #     bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        grid_x = np.tile(np.tile(np.linspace(0, in_w-1, in_w), (in_w, 1)), (bs * self.num_anchors, 1, 1)).reshape(x.shape)
        grid_y = np.tile(np.tile(np.linspace(0, in_h-1, in_h), (in_h, 1)).T, (bs * self.num_anchors, 1, 1)).reshape(y.shape)


        # Calculate anchor w, h
        # anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        # anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        # anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        # anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        anchor_w = np.array(scaled_anchors)[:, 0].reshape(1, self.num_anchors, 1, 1)
        anchor_h = np.array(scaled_anchors)[:, 1].reshape(1, self.num_anchors, 1, 1)
        anchor_w = np.tile(anchor_w, (bs, 1, in_h, in_w))
        anchor_h = np.tile(anchor_h, (bs, 1, in_h, in_w))


        # Add offset and scale with anchors
        pred_boxes = np.empty_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = exp(w) * anchor_w
        pred_boxes[..., 3] = exp(h) * anchor_h


        # Results
        # _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        _scale = np.array([stride_w, stride_h] * 2)

        # output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
        #                     conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
        output = np.concatenate((
            (pred_boxes.reshape(bs, -1, 4) * _scale),
            conf.reshape(bs, -1, 1),
            pred_cls.reshape(bs, -1, self.num_classes)
        ), axis=-1)
        
        return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  np.maximum(b1_x1, b2_x1)
    inter_rect_y1 =  np.maximum(b1_y1, b2_y1)
    inter_rect_x2 =  np.minimum(b1_x2, b2_x2)
    inter_rect_y2 =  np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.maximum(0, inter_rect_x2 - inter_rect_x1 + 1) * np.maximum(0, inter_rect_y2 - inter_rect_y1 + 1)

    # Union Area
    b1_area = abs((b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1))
    b2_area = abs((b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1))

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    
    # box_corner = prediction.new(prediction.shape)
    box_corner = np.copy(prediction)

    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        print("Entered NMS loop!!")

        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        print(image_pred.shape) # (17, 85)

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
            
        # Get score and class with highest confidence
        # class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        class_pred = class_pred[:, np.newaxis]  # Add a new axis to mimic keepdims=True

        # print(class_pred, class_conf)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        detections = np.concatenate((image_pred[:, :5], class_conf.astype(np.float32), class_pred.astype(np.float32)), axis=1)

        # Iterate through all predicted classes
        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]

            # Sort the detections by maximum objectness confidence
            # _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            # detections_class = detections_class[conf_sort_index]
            conf_sort_index = np.argsort(detections_class[:, 4])[::-1]
            detections_class = detections_class[conf_sort_index]

            # Perform non-maximum suppression
            max_detections = []
            while detections_class.shape[0]:
                # Get detection with highest confidence and save as max detection
                # max_detections.append(detections_class[0].unsqueeze(0))
                max_detections.append(detections_class[0].reshape(1, -1))

                # Stop if we're at the last detection
                if detections_class.shape[0] == 1:
                    break

                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])

                # Remove detections with IoU >= NMS threshold
                # detections_class = detections_class[1:][ious < nms_thres]

                # Find indices of detections with IoU < NMS threshold
                keep_indices = np.where(ious < nms_thres)[0]

                # Filter detections to keep only those with IoU < NMS threshold
                detections_class = detections_class[1:][keep_indices]

            max_detections = np.concatenate(max_detections, axis=0)

            # Add max detections to outputs
            # output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            if output[image_i] is None:
                output[image_i] = max_detections
            else:
                output[image_i] = np.concatenate((output[image_i], max_detections), axis=0)

    return output