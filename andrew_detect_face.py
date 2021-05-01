#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './output_model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class BoundingBox(object):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)
        
OBJECT_CLASSES = ['background', 'face']
class Inference(object):
    def __init__(self, object_class, score, bounding_box):
        self.object_class = OBJECT_CLASSES[int(object_class)]
        self.score = score
        self.bounding_box = bounding_box

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class ImageInferences(object):
    def __init__(self, image_id, inferences):
        self.image_id = image_id
        self.inferences = inferences

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        [w, h] = image.shape[:2]
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # print(len(image_np_expanded))
        # print(len(image_np_expanded[0]))
        # print(len(image_np_expanded[0][0]))
        # print(len(image_np_expanded[0][0][0]))

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        # num_detections = num_detections[0]
        # classes = classes[0]
        # scores = scores[0]
        # boxes = [[box[0] * w, box[1]*h, box[2]*w, box[3]*h] for box in boxes[0]]

        num_detections = int(num_detections[0])
        classes = classes[0]
        scores = scores[0]
        boxes = [BoundingBox(box[0] * w, box[2]*w, box[1]*h, box[3]*h) for box in boxes[0]]
        
        inferences = []
        for i in range(num_detections):
            if scores[i] < 0.01:
                continue
            inferences.append(Inference(int(classes[i]), float(scores[i]), boxes[i]))

        return inferences
        # return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print """usage:%s (cameraID | filename)
    #             Dectect faces in the video
    #             example:
    #             %s 0
    #             """ % (sys.argv[0], sys.argv[0])
    #     exit(1)

    # try:
    #     camID = int(sys.argv[1])
    # except:
    #     camID = sys.argv[1]
    
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    path = "/mnt/linux-only/projects/face-detection-ssd-mobilenet/data/WIDER_train/images/12--Group/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_994.jpg"
    image_id = os.path.basename(path).split(".", 1)[0]
    print(image_id)
    image = cv2.imread(path)

    [h, w] = image.shape[:2]
    # image = cv2.flip(image, 1)

    inferences = tDetector.run(image)
    # print([str(inf) for inf in inferences[:10]])
    image_inferences = ImageInferences(image_id, inferences)
    print(image_inferences)

    '''
    # print("num_detections: ", num_detections[0])
    # print("scores: ", scores[0][:10])
    # print("classes: ", classes[0][:10])
    # print("boxes: ", [[box[0]*w, box[1]*h, box[2]*w, box[3]*h] for box in boxes[0][:10]])
    # print("boxes: ", [[box[0]*h, box[1]*w, box[2]*h, box[3]*w] for box in boxes[0][:10]])

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

    # cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
    cv2.imwrite("testfile.png", image)
    '''
