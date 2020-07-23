import numpy as np
import os
import logging as log
import cv2
from PIL import Image
from openvino.inference_engine import IECore
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
import sys

# anchor configuration
# feature_map_sizes = [[20, 20], [10, 10], [5, 5], [3, 3], [2, 2]]  # for input size 160
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]  # for input size 260
# feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]  # for input size 360
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


class MaskNetwork:

    def __init__(self, device="MYRIAD", cpu_extension=False):
        self._exec_net = None
        self._input_blob = None

        self._start_openvino(device, cpu_extension)

    def _start_openvino(self, device, cpu_extension):
        '''
        Initializes OpenVino and loads network
        
        device: Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is
                acceptable. The sample will look for a suitable plugin for device specified.
        '''

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_xml = os.path.join(root_dir, "models", "openvino", "face_mask_detection.xml")
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Creating Inference Engine")
        ie = IECore()
        if cpu_extension and 'CPU' in device:
            ie.add_extension(cpu_extension, "CPU")

        # Read IR (load the model)
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = ie.read_network(model=model_xml, weights=model_bin)

        if "CPU" in device:
            supported_layers = ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        log.info("Preparing input blobs")
        self._input_blob = next(iter(net.inputs))
        # out_blob = next(iter(net.outputs))

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self._exec_net = ie.load_network(network=net, device_name=device)

    def openvino_infer(self, image,
                       conf_thresh=0.5,
                       iou_thresh=0.4,
                       target_shape=(260, 260),
                       draw_result=True,
                       show_result=False
                       ):
        height, width, _ = image.shape

        # resize & adjust input
        image_resized = cv2.resize(image, target_shape)
        image_np = image_resized / 255.0
        image_exp = np.expand_dims(image_np, axis=0)
        image_transposed = image_exp.transpose((0, 3, 1, 2))  # Change data layout from HWC to CHW

        res = self._exec_net.infer(inputs={self._input_blob: image_transposed})

        # result conversion
        y_cls_output = res['cls_branch_concat']
        y_bboxes_output = res['loc_branch_concat']
        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]

        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # To speed up, do single class NMS, not multiple classes NMS.
        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                     bbox_max_scores,
                                                     conf_thresh=conf_thresh,
                                                     iou_thresh=iou_thresh,
                                                     )

        # build detections for output (and draw if needed)
        output_info = []
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            if draw_result:
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

        if show_result:
            Image.fromarray(image).show()

        return output_info
