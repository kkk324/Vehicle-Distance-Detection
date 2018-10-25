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
import math
import settings


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
	default='model_data/imagenet_classes.txt')
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

def area(bbox):
	return float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))


def get_center_shift(coeffs, img_size, pixels_per_meter):
	return np.polyval(coeffs, img_size[1]/pixels_per_meter[1]) - (img_size[0]//2)/pixels_per_meter[0]

def get_curvature(coeffs, img_size, pixels_per_meter):
	return ((1 + (2*coeffs[0]*img_size[1]/pixels_per_meter[1] + coeffs[1])**2)**1.5) / np.absolute(2*coeffs[0])


class LaneLineFinder:
	def __init__(self, img_size, pixels_per_meter, center_shift):
		self.found = False
		self.poly_coeffs = np.zeros(3, dtype=np.float32)
		self.coeff_history = np.zeros((3, 7), dtype=np.float32)
		self.img_size = img_size
		self.pixels_per_meter = pixels_per_meter
		self.line_mask = np.ones((img_size[1], img_size[0]), dtype=np.uint8)
		self.other_line_mask = np.zeros_like(self.line_mask)
		self.line = np.zeros_like(self.line_mask)
		self.num_lost = 0
		self.still_to_find = 1
		self.shift = center_shift
		self.first = True
		self.stddev = 0

	def reset_lane_line(self):
		self.found = False
		self.poly_coeffs = np.zeros(3, dtype=np.float32)
		self.line_mask[:] = 1
		self.first = True

	def one_lost(self):
		self.still_to_find = 5
		if self.found:
			self.num_lost += 1
			if self.num_lost >= 7:
				self.reset_lane_line()

	def one_found(self):
		self.first = False
		self.num_lost = 0
		if not self.found:
			self.still_to_find -= 1
			if self.still_to_find <= 0:
				self.found = True

	def fit_lane_line(self, mask):
		y_coord, x_coord = np.where(mask)
		y_coord = y_coord.astype(np.float32) / self.pixels_per_meter[1]
		x_coord = x_coord.astype(np.float32) / self.pixels_per_meter[0]
		if len(y_coord) <= 150:
			coeffs = np.array([0, 0, (self.img_size[0] // 2) / self.pixels_per_meter[0] + self.shift], dtype=np.float32)
		else:
			coeffs, v = np.polyfit(y_coord, x_coord, 2, rcond=1e-16, cov=True)
			self.stddev = 1 - math.exp(-5 * np.sqrt(np.trace(v)))

		self.coeff_history = np.roll(self.coeff_history, 1)

		if self.first:
			self.coeff_history = np.reshape(np.repeat(coeffs, 7), (3, 7))
		else:
			self.coeff_history[:, 0] = coeffs

		value_x = get_center_shift(coeffs, self.img_size, self.pixels_per_meter)
		curve = get_curvature(coeffs, self.img_size, self.pixels_per_meter)

		if (self.stddev > 0.95) | (len(y_coord) < 150) | (math.fabs(value_x - self.shift) > math.fabs(0.5 * self.shift)) \
				| (curve < 30):

			self.coeff_history[0:2, 0] = 0
			self.coeff_history[2, 0] = (self.img_size[0] // 2) / self.pixels_per_meter[0] + self.shift
			self.one_lost()
		else:
			self.one_found()

		self.poly_coeffs = np.mean(self.coeff_history, axis=1)

	def get_line_points(self):
		y = np.array(range(0, self.img_size[1] + 1, 10), dtype=np.float32) / self.pixels_per_meter[1]
		x = np.polyval(self.poly_coeffs, y) * self.pixels_per_meter[0]
		y *= self.pixels_per_meter[1]
		return np.array([x, y], dtype=np.int32).T

	def get_other_line_points(self):
		pts = self.get_line_points()
		pts[:, 0] = pts[:, 0] - 2 * self.shift * self.pixels_per_meter[0]
		return pts

	def find_lane_line(self, mask, reset=False):
		n_segments = 16
		window_width = 30
		step = self.img_size[1] // n_segments

		if reset or (not self.found and self.still_to_find == 5) or self.first:
			self.line_mask[:] = 0
			n_steps = 4
			window_start = self.img_size[0] // 2 + int(self.shift * self.pixels_per_meter[0]) - 3 * window_width
			window_end = window_start + 6 * window_width
			sm = np.sum(mask[self.img_size[1] - 4 * step:self.img_size[1], window_start:window_end], axis=0)
			sm = np.convolve(sm, np.ones((window_width,)) / window_width, mode='same')
			argmax = window_start + np.argmax(sm)
			shift = 0
			for last in range(self.img_size[1], 0, -step):
				first_line = max(0, last - n_steps * step)
				sm = np.sum(mask[first_line:last, :], axis=0)
				sm = np.convolve(sm, np.ones((window_width,)) / window_width, mode='same')
				window_start = min(max(argmax + int(shift) - window_width // 2, 0), self.img_size[0] - 1)
				window_end = min(max(argmax + int(shift) + window_width // 2, 0 + 1), self.img_size[0])
				new_argmax = window_start + np.argmax(sm[window_start:window_end])
				new_max = np.max(sm[window_start:window_end])
				if new_max <= 2:
					new_argmax = argmax + int(shift)
					shift = shift / 2
				if last != self.img_size[1]:
					shift = shift * 0.25 + 0.75 * (new_argmax - argmax)
				argmax = new_argmax
				cv2.rectangle(self.line_mask, (argmax - window_width // 2, last - step),
							  (argmax + window_width // 2, last),
							  1, thickness=-1)
		else:
			self.line_mask[:] = 0
			points = self.get_line_points()
			if not self.found:
				factor = 3
			else:
				factor = 2
			cv2.polylines(self.line_mask, [points], 0, 1, thickness=int(factor * window_width))

		self.line = self.line_mask * mask
		self.fit_lane_line(self.line)
		self.first = False
		if not self.found:
			self.line_mask[:] = 1
		points = self.get_other_line_points()
		self.other_line_mask[:] = 0
		cv2.polylines(self.other_line_mask, [points], 0, 1, thickness=int(5 * window_width))


# class that finds the whole lane
class LaneFinder:
	def __init__(self, img_size, warped_size, cam_matrix, dist_coeffs, transform_matrix, pixels_per_meter,
				 warning_icon):
		self.found = False
		self.cam_matrix = cam_matrix
		self.dist_coeffs = dist_coeffs
		self.img_size = img_size
		self.warped_size = warped_size
		self.mask = np.zeros((warped_size[1], warped_size[0], 3), dtype=np.uint8)
		self.roi_mask = np.ones((warped_size[1], warped_size[0], 3), dtype=np.uint8)
		self.total_mask = np.zeros_like(self.roi_mask)
		self.warped_mask = np.zeros((self.warped_size[1], self.warped_size[0]), dtype=np.uint8)
		self.M = transform_matrix
		self.count = 0
		self.left_line = LaneLineFinder(warped_size, pixels_per_meter, -1.8288)  # 6 feet in meters
		self.right_line = LaneLineFinder(warped_size, pixels_per_meter, 1.8288)
		if (warning_icon is not None):
			self.warning_icon = np.array(mpimg.imread(warning_icon) * 255, dtype=np.uint8)
		else:
			self.warning_icon = None

	def undistort(self, img):
		return cv2.undistort(img, self.cam_matrix, self.dist_coeffs)

	def warp(self, img):
		return cv2.warpPerspective(img, self.M, self.warped_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

	def unwarp(self, img):
		return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.WARP_FILL_OUTLIERS +
																	 cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

	def equalize_lines(self, alpha=0.9):
		mean = 0.5 * (self.left_line.coeff_history[:, 0] + self.right_line.coeff_history[:, 0])
		self.left_line.coeff_history[:, 0] = alpha * self.left_line.coeff_history[:, 0] + \
											 (1 - alpha) * (mean - np.array([0, 0, 1.8288], dtype=np.uint8))
		self.right_line.coeff_history[:, 0] = alpha * self.right_line.coeff_history[:, 0] + \
											  (1 - alpha) * (mean + np.array([0, 0, 1.8288], dtype=np.uint8))

	def find_lane(self, img, distorted=True, reset=False):
		# undistort, warp, change space, filter
		if distorted:
			img = self.undistort(img)
		if reset:
			self.left_line.reset_lane_line()
			self.right_line.reset_lane_line()

		img = self.warp(img)
		img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		img_hls = cv2.medianBlur(img_hls, 5)
		img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
		img_lab = cv2.medianBlur(img_lab, 5)

		big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
		small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

		greenery = (img_lab[:, :, 2].astype(np.uint8) > 130) & cv2.inRange(img_hls, (0, 0, 50), (35, 190, 255))

		road_mask = np.logical_not(greenery).astype(np.uint8) & (img_hls[:, :, 1] < 250)
		road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, small_kernel)
		road_mask = cv2.dilate(road_mask, big_kernel)

		img2, contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		biggest_area = 0
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > biggest_area:
				biggest_area = area
				biggest_contour = contour
		road_mask = np.zeros_like(road_mask)
		cv2.fillPoly(road_mask, [biggest_contour], 1)

		self.roi_mask[:, :, 0] = (self.left_line.line_mask | self.right_line.line_mask) & road_mask
		self.roi_mask[:, :, 1] = self.roi_mask[:, :, 0]
		self.roi_mask[:, :, 2] = self.roi_mask[:, :, 0]

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
		black = cv2.morphologyEx(img_lab[:, :, 0], cv2.MORPH_TOPHAT, kernel)
		lanes = cv2.morphologyEx(img_hls[:, :, 1], cv2.MORPH_TOPHAT, kernel)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
		lanes_yellow = cv2.morphologyEx(img_lab[:, :, 2], cv2.MORPH_TOPHAT, kernel)

		self.mask[:, :, 0] = cv2.adaptiveThreshold(black, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)
		self.mask[:, :, 1] = cv2.adaptiveThreshold(lanes, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
		self.mask[:, :, 2] = cv2.adaptiveThreshold(lanes_yellow, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
												   13, -1.5)
		self.mask *= self.roi_mask
		small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		self.total_mask = np.any(self.mask, axis=2).astype(np.uint8)
		self.total_mask = cv2.morphologyEx(self.total_mask.astype(np.uint8), cv2.MORPH_ERODE, small_kernel)

		left_mask = np.copy(self.total_mask)
		right_mask = np.copy(self.total_mask)
		if self.right_line.found:
			left_mask = left_mask & np.logical_not(self.right_line.line_mask) & self.right_line.other_line_mask
		if self.left_line.found:
			right_mask = right_mask & np.logical_not(self.left_line.line_mask) & self.left_line.other_line_mask
		self.left_line.find_lane_line(left_mask, reset)
		self.right_line.find_lane_line(right_mask, reset)
		self.found = self.left_line.found and self.right_line.found

		if self.found:
			self.equalize_lines(0.875)

	def draw_lane_weighted(self, img, thickness=5, alpha=0.8, beta=1, gamma=0):
		left_line = self.left_line.get_line_points()
		right_line = self.right_line.get_line_points()
		both_lines = np.concatenate((left_line, np.flipud(right_line)), axis=0)
		lanes = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)
		if self.found:
			cv2.fillPoly(lanes, [both_lines.astype(np.int32)], (0, 255, 0))
			cv2.polylines(lanes, [left_line.astype(np.int32)], False, (255, 0, 0), thickness=thickness)
			cv2.polylines(lanes, [right_line.astype(np.int32)], False, (0, 0, 255), thickness=thickness)
			cv2.fillPoly(lanes, [both_lines.astype(np.int32)], (0, 255, 0))
			mid_coef = 0.5 * (self.left_line.poly_coeffs + self.right_line.poly_coeffs)
			curve = get_curvature(mid_coef, img_size=self.warped_size, pixels_per_meter=self.left_line.pixels_per_meter)
			shift = get_center_shift(mid_coef, img_size=self.warped_size,
									 pixels_per_meter=self.left_line.pixels_per_meter)
			cv2.putText(img, "Road curvature: {:6.2f}m".format(curve), (420, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
						thickness=5, color=(255, 255, 255))
			cv2.putText(img, "Road curvature: {:6.2f}m".format(curve), (420, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
						thickness=3, color=(0, 0, 0))
			cv2.putText(img, "Car position: {:4.2f}m".format(shift), (460, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
						thickness=5, color=(255, 255, 255))
			cv2.putText(img, "Car position: {:4.2f}m".format(shift), (460, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
						thickness=3, color=(0, 0, 0))
		else:
			warning_shape = self.warning_icon.shape
			corner = (10, (img.shape[1] - warning_shape[1]) // 2)
			patch = img[corner[0]:corner[0] + warning_shape[0], corner[1]:corner[1] + warning_shape[1]]
			patch[self.warning_icon[:, :, 3] > 0] = self.warning_icon[self.warning_icon[:, :, 3] > 0, 0:3]
			img[corner[0]:corner[0] + warning_shape[0], corner[1]:corner[1] + warning_shape[1]] = patch
			cv2.putText(img, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
						thickness=5, color=(255, 255, 255))
			cv2.putText(img, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
						thickness=3, color=(0, 0, 0))
		lanes_unwarped = self.unwarp(lanes)
		return cv2.addWeighted(img, alpha, lanes_unwarped, beta, gamma)

	def process_image(self, img, reset=False, show_period=10, blocking=False):
		self.find_lane(img, reset=reset)
		lane_img = self.draw_lane_weighted(img)
		self.count += 1
		if show_period > 0 and (self.count % show_period == 1 or show_period == 1):
			start = 231
			plt.clf()
			for i in range(3):
				plt.subplot(start + i)
				plt.imshow(lf.mask[:, :, i] * 255, cmap='gray')
				plt.subplot(234)
			plt.imshow((lf.left_line.line + lf.right_line.line) * 255)

			ll = cv2.merge((lf.left_line.line, lf.left_line.line * 0, lf.right_line.line))
			lm = cv2.merge((lf.left_line.line_mask, lf.left_line.line * 0, lf.right_line.line_mask))
			plt.subplot(235)
			plt.imshow(lf.roi_mask * 255, cmap='gray')
			plt.subplot(236)
			plt.imshow(lane_img)
			if blocking:
				plt.show()
			else:
				plt.draw()
				plt.pause(0.000001)
		return lane_img




def _main(args, lf):

	### Video
	video_path = '/home/crke/Work/YAD2K/input_videos/tw_test_short.mp4' #'/home/crke/Work/YAD2K/input_videos/harder_challenge_video.mp4' #'/home/crke/Work/YAD2K/input_videos/project_video.mp4' #'/home/crke/Work/YAD2K/input_videos/challenge_video.mp4' #'
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

	yolo_model = load_model(model_path)
	yolo_model.summary()
	
	# Verify model, anchors, and classes are compatible
	num_classes = len(class_names)
	num_anchors = len(anchors)
	# TODO: Assumes dim ordering is channel last
	model_output_channels = yolo_model.layers[-1].output_shape[-1]

	assert model_output_channels == (num_classes ), \
		'Mismatch between model and given anchor and class sizes. ' \
		'Specify matching anchors and classes with --anchors_path and ' \
		'--classes_path flags.'

	# assert model_output_channels == num_anchors * (num_classes + 5), \
	# 	'Mismatch between model and given anchor and class sizes. ' \
	# 	'Specify matching anchors and classes with --anchors_path and ' \
	# 	'--classes_path flags.'

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

########################################################################
	# # Image for debug
	# image_file = 'test4.jpg'
	# image_shape = (720., 1280.)
	#
	# frame, image_data = preprocess_image("images/" + image_file, (int(MODEL_SIZE[0]), int(MODEL_SIZE[1])))
	# # frame       1280x720
	# # image_data  608x608
	#
	# out_boxes, out_scores, out_classes = sess.run(
	# 	[boxes, scores, classes],
	# 	feed_dict={
	# 		yolo_model.input: image_data,
	# 		input_image_shape: [(image_shape[0]), (image_shape[1])],
	# 		K.learning_phase(): 0
	# 	})
	#
	#
	# fframe = np.array(frame)
	# fframe = lf.process_image(fframe, True, show_period=1, blocking=False)
	# frame = Image.fromarray(fframe)
	#
	# l = len(out_boxes)
	# distance = np.zeros(shape=(l ,1))
	# if not len(out_boxes) == 0:
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
	# 	#print('No Car')
	#
	# frame.save(os.path.join('out', image_file), quality=90)

	### END
########################################################################


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
			# out_boxes is already recale to original size
			duration = time.time() - t0
			print('duration', duration)
			print('fps', 1 / duration)
			print('out_boxes', out_boxes)


			###

			# fframe = np.array(frame)
			# fframe = lf.process_image(fframe, False, show_period=40, blocking=False)
			# frame = Image.fromarray(fframe)

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
				#print('No Car')

			pix = np.array(frame)
			pix = pix[:,:,::-1]
			outVideo.write(pix)
		# Break the loop
		else:
			break

	cap.release()
	outVideo.release()
	sess.close()
	print("Finish video convert !!!")



if __name__ == '__main__':
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

	lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, cam_matrix, dist_coeffs,
					perspective_transform, pixels_per_meter, "warning.png")

	_main(parser.parse_args(), lf)
