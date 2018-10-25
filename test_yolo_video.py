#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random
import time,cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from yad2k.models.keras_yolo import yolo_eval, yolo_head
from settings import *
from yolo_utils import *

import tensorflow as tf
import pickle

parser = argparse.ArgumentParser(
	description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
	'model_path',
	help='path to h5 model file containing body'
	'of a YOLO_v2 model')
parser.add_argument(
	'-a',
	'--anchors_path',
	help='path to anchors file, defaults to yolo_anchors.txt',
	default='model_data/yolo_anchors.txt')
parser.add_argument(
	'-c',
	'--classes_path',
	help='path to classes file, defaults to coco_classes.txt',
	default='model_data/coco_classes.txt')
parser.add_argument(
	'-t',
	'--test_path',
	help='path to directory of test images, defaults to images/',
	default='images')
parser.add_argument(
	'-o',
	'--output_path',
	help='path to output test images, defaults to images/out',
	default='images/out')
parser.add_argument(
	'-s',
	'--score_threshold',
	type=float,
	help='threshold for bounding box scores, default .3',
	default=.3)
parser.add_argument(
	'-iou',
	'--iou_threshold',
	type=float,
	help='threshold for non max suppression IOU, default .5',
	default=.5)

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
	"""Filters YOLO boxes by thresholding on object and class confidence.

	Arguments:
	box_confidence -- tensor of shape (19, 19, 5, 1)
	boxes -- tensor of shape (19, 19, 5, 4)
	box_class_probs -- tensor of shape (19, 19, 5, 80)
	threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

	Returns:
	scores -- tensor of shape (None,), containing the class probability score for selected boxes
	boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
	classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

	Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
	For example, the actual output size of scores would be (10,) if there are 10 boxes.
	"""

	# Step 1: Compute box scores
	box_scores = box_confidence * box_class_probs  # [19, 19, 5, 1] * [19, 19, 5, 80] = [19, 19, 5, 80]

	# Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
	box_classes = K.argmax(box_scores, axis=-1)
	box_class_scores = K.max(box_scores, axis=-1, keepdims=False)

	# Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	# same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
	filtering_mask = box_class_scores >= threshold

	# Step 4: Apply the mask to scores, boxes and classes
	scores = tf.boolean_mask(box_class_scores, filtering_mask)
	boxes = tf.boolean_mask(boxes, filtering_mask)
	classes = tf.boolean_mask(box_classes, filtering_mask)

	return scores, boxes, classes

def preprocess_image(img_path, model_image_size):
	image_type = imghdr.what(img_path)
	image = Image.open(img_path)
	resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
	image_data = np.array(resized_image, dtype='float32')
	image_data /= 255.
	image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
	return image, image_data


def preprocess_frame(frame, model_image_size):
	frame = frame[:,:,::-1]
	image = frame
	frame = cv2.resize(frame, (model_image_size[0], model_image_size[1]))
	image_data = np.array(frame, dtype='float32')
	# image = Image.fromarray(np.uint8(cm.gist_earth(frame) * 255))
	#image = Image.fromarray(cm.gist_earth(frame, bytes=True))
	image = Image.fromarray(image.astype('uint8'), 'RGB')
	#image = Image.fromarray(frame)
	image_data /= 255.
	image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
	return image, image_data

def calculate_position(bbox, transform_matrix, warped_size, pix_per_meter):

	if len(bbox) == 0:
		print('Nothing')
	else:
		pos = np.array((bbox[1]/2+bbox[3]/2, bbox[2])).reshape(1, 1, -1)
		dst = cv2.perspectiveTransform(pos, transform_matrix).reshape(-1, 1)
		return np.array((warped_size[1]-dst[1])/pix_per_meter[1])



def _main(args):


	with open(CALIB_FILE_NAME, 'rb') as f:
		calib_data = pickle.load(f)
		cam_matrix = calib_data["cam_matrix"]
		dist_coeffs = calib_data["dist_coeffs"]
		img_size = calib_data["img_size"]


	with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
		perspective_data = pickle.load(f)

	perspective_transform = perspective_data["perspective_transform"]
	pixels_per_meter = perspective_data['pixels_per_meter']
	orig_points = perspective_data["orig_points"]


	### Video
	video_path = '/home/crke/Work/YAD2K/input_videos/challenge_video.mp4' #'/home/crke/Work/YAD2K/input_videos/harder_challenge_video.mp4' #'/home/crke/Work/YAD2K/input_videos/project_video.mp4'
	output_path = '/home/crke/Work/YAD2K/output_videos/'
	output_Video = os.path.basename(video_path)
	output_Video = os.path.join(output_path, output_Video)

	cap = cv2.VideoCapture(video_path)
	FrameCnt = 0
	fps = cap.get(cv2.CAP_PROP_FPS)
	FrameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	isModelSize = False
	if Width == 608 and Height == 608:
		isModelSize = True

	print("Video Info:")
	print("Input: ", video_path)
	print("FPS: ", fps)
	print("FrameNum: ", FrameNum)
	print("Width: ", Width)
	print("Height: ", Height)
	print("Output: ", output_Video)

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # DIVX, XVID, MJPG, X264, WMV1, WMV2
	outVideo = cv2.VideoWriter(output_Video, fourcc, fps, (VIDEO_SIZE[0], VIDEO_SIZE[1]))

	###

	model_path = os.path.expanduser(args.model_path)
	assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
	anchors_path = os.path.expanduser(args.anchors_path)
	classes_path = os.path.expanduser(args.classes_path)
	test_path = os.path.expanduser(args.test_path)
	output_path = os.path.expanduser(args.output_path)

	if not os.path.exists(output_path):
		print('Creating output path {}'.format(output_path))
		os.mkdir(output_path)

	sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

	with open(classes_path) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]

	with open(anchors_path) as f:
		anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		anchors = np.array(anchors).reshape(-1, 2)

	yolo_model = load_model(model_path)

	# Verify model, anchors, and classes are compatible
	num_classes = len(class_names)
	num_anchors = len(anchors)
	# TODO: Assumes dim ordering is channel last
	model_output_channels = yolo_model.layers[-1].output_shape[-1]
	assert model_output_channels == num_anchors * (num_classes + 5), \
		'Mismatch between model and given anchor and class sizes. ' \
		'Specify matching anchors and classes with --anchors_path and ' \
		'--classes_path flags.'
	print('{} model, anchors, and classes loaded.'.format(model_path))

	# Check if model is fully convolutional, assuming channel last order.
	model_image_size = yolo_model.layers[0].input_shape[1:3]
	is_fixed_size = model_image_size != (None, None)

	# Generate colors for drawing bounding boxes.
	hsv_tuples = [(x / len(class_names), 1., 1.)
				  for x in range(len(class_names))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
			colors))
	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.

	# Generate output tensor targets for filtered bounding boxes.
	# TODO: Wrap these backend operations with Keras layers.
	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
	input_image_shape = K.placeholder(shape=(2, ))
	boxes, scores, classes = yolo_eval(
		yolo_outputs,
		input_image_shape,
		score_threshold=args.score_threshold,
		iou_threshold=args.iou_threshold)


	# Image for debug
	# image_file = 'test4.jpg'
	# image_shape = (720., 1280.)
	#
	# frame, image_data = preprocess_image("images/" + image_file, (int(MODEL_SIZE[0]), int(MODEL_SIZE[1])))
	# out_boxes, out_scores, out_classes = sess.run(
	# 	[boxes, scores, classes],
	# 	feed_dict={
	# 		yolo_model.input: image_data,
	# 		input_image_shape: [(image_shape[0]), (image_shape[1])],
	# 		K.learning_phase(): 0
	# 	})
	#
	#
	# distance = np.zeros(shape=(3,1))
	# if not len(out_boxes) == 0:
	# 	l = len(out_boxes)
	# 	for i in range(l):
	# 		distance[i] = calculate_position(bbox=out_boxes[i],
	# 								  transform_matrix=perspective_transform,
	# 								  warped_size=UNWARPED_SIZE,
	# 								  pix_per_meter=pixels_per_meter)
	#
	# 	print('RPOS', distance)
	# 	draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors, distance)
	#
	# else:
	# 	distance = []
	# 	print('No Car')
	#
	# frame.save(os.path.join('out', image_file), quality=90)
    ### END

	image_shape = (720., 1280.)
	# Read until video is completed
	while (cap.isOpened()):

		ret, frame = cap.read()

		batch = 1
		if ret == True:
			index = (FrameCnt + 1) % batch

			frame, image_data = preprocess_frame(frame, (int(MODEL_SIZE[0]), int(MODEL_SIZE[1])))

			t0 = time.time()
			out_boxes, out_scores, out_classes = sess.run(
				[boxes, scores, classes],
				feed_dict={
					yolo_model.input: image_data,
					input_image_shape: [(image_shape[0]), (image_shape[1])],
					K.learning_phase(): 0
				})

			duration = time.time() - t0
			print('duration', duration)
			print('fps', 1 / duration)
			print('out_boxes', out_boxes)


			###
			l = len(out_boxes)
			distance = np.zeros(shape=(l, 1))
			if not len(out_boxes) == 0:
				for i in range(l):
					distance[i] = calculate_position(bbox=out_boxes[i],
					                                 transform_matrix=perspective_transform,
					                                 warped_size=UNWARPED_SIZE,
					                                 pix_per_meter=pixels_per_meter)

				print('RPOS', distance)
				draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors, distance)

			else:
				distance = []
				print('No Car')


			draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors, distance)

			#resized_image = frame.resize(tuple(reversed((VIDEO_SIZE[1], VIDEO_SIZE[0]))), Image.BICUBIC)
			#resized_image.save('/home/crke/Work/YAD2K/1.jpg' , quality=90)
			pix = np.array(frame)
			pix = pix[:,:,::-1]
			#pix.resize((Width, Height))
			outVideo.write(pix)
		# Break the loop
		else:
			break

	cap.release()
	outVideo.release()
	sess.close()
	print("Finish video convert !!!")

if __name__ == '__main__':

	_main(parser.parse_args())
