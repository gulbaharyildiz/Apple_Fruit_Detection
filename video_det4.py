# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:46:53 2021

@author: aad
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:19:13 2021

@author: aad
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from six import BytesIO
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
 
# while "models" in pathlib.Path.cwd().parts:
#     os.chdir('..')
 
# def load_model():
#   # base_url = 'http://download.tensorflow.org/models/object_detection/'
#   # model_file = model_name + '.tar.gz'
#   # model_dir = tf.keras.utils.get_file(
#   #   fname=model_name, 
#   #   origin=base_url + model_file,
#   #   untar=True)
 
#   # model_dir = pathlib.Path(model_dir)/"saved_model"
#   model_dir='E:/projects\object_detection/apple_detection/apple_code2/models/research/object_detection/inference_graph3/saved_model'
#   model = tf.saved_model.load(str(model_dir))
#   return model
 
PATH_TO_LABELS = 'E:/projects\object_detection/apple_detection/apple_code2/models/research/object_detection/training6/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
 
#model_name = 'ssd_inception_v2_coco_2017_11_17'
tf.keras.backend.clear_session()
detection_model= tf.saved_model.load(f'E:/projects/object_detection/apple_detection/apple_code2/models/research/object_detection/inference_graph6/saved_model')
 
def run_inference_for_single_image(model, image):
  #image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]
 
  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
 
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
 
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
     
  return output_dict

# def show_inference(model, frame):
#   #take the frame from webcam feed and convert that to array
#   image_np = np.array(frame)
#   # Actual detection.
     
#   output_dict = run_inference_for_single_image(model, image_np)
#   # Visualization of the results of a detection.
#   vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks_reframed', None),
#       use_normalized_coordinates=True,
#       line_thickness=1)
 
#   return(image_np)

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Now we open the webcam and start detecting objects
import cv2
#video_capture = cv2.VideoCapture(0)
video_capture=cv2.VideoCapture('E:/projects/object_detection/apple_detection/apple_code2/models/research/object_detection/Apple farm 3D video of kashmir.mp4')
#video_capture=cv2.VideoCapture('E:/projects/object_detection/apple_detection/apple_code2/models/research/object_detection/videoplayback_Trim.mp4')
re,frame = video_capture.read()
height,width,layers=frame.shape

fourcc=cv2.VideoWriter_fourcc(*'XVID')
video=cv2.VideoWriter('output.avi',fourcc,20.0,(width,height))
while True:
    # Capture frame-by-frame
    re,frame = video_capture.read()
    cv2.imwrite("frame.jpg", frame)
    path='E:/projects/object_detection/apple_detection/apple_code2/models/research/object_detection/frame.jpg'
    image_np = load_image_into_numpy_array(path)
    # img = cv2.imread(path) 
    # image_np = np.array(img).reshape((height, width, 3)).astype(np.uint8)
    output_dict = run_inference_for_single_image(detection_model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=1)
    #display(Image.fromarray(image_np))
    final_score = np.squeeze(output_dict['detection_scores'])    
    count = 0
    for i in range(len(final_score)):
      if final_score[i] is None or final_score[i] > 0.5:
        count = count + 1
    print('Total Number of detected apples in Image: ',count)
    image_np = cv2.putText(image_np, 'Total Number of detected apples:{} '.format(count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.7, ((0, 0, 255) ), 2, cv2.LINE_AA) 
    cv2.imshow('object detection', image_np)
    video.write(image_np)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
video.release()
