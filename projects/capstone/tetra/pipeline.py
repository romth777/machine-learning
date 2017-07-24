import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import pickle
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

import ImageProcessUtils

from generate_tracklet import *
import re


class Pipeline:
    def __init__(self):
        self.ipu = ImageProcessUtils.ImageProcessUtils()
        self.do_kb = True
        self.do_yolov2 = False
        self.model = None
        self.annotate = None
        self.target = ["car", "person"]

        if self.do_kb:
            from KittiBox import annotate
            self.annotate = annotate.annotate()
        if self.do_yolov2:
            from YAD2K import annotate
            self.annotate = annotate.annotate()

    # reset function clear the data of previous analysis
    def reset(self):
        pass

    # Main pipeline process of this project
    def pipeline_kb(self, img):
        # convert dtype for uint8 for processing
        img = img.astype(np.uint8)

        # apply kittibox
        out_img, pred_boxes = self.annotate.make_annotate(img, threshold=0.5)

        #windowed_img = self.ipu.draw_boxes(img, pred_boxes, color=(0, 0, 255), thick=6)
        #plt.imshow(windowed_img)
        #plt.show()

        # stitch windows to centeroid and filter out false positive with heatmap
        heatmap = np.zeros_like(img[:, :, 0])
        heat = self.ipu.add_heat(heatmap, pred_boxes)
        #plt.imshow(heat)
        #plt.show()

        self.ipu.apply_threshold(heat, 100000, 3)
        labels = label(heatmap)
        draw_img = self.ipu.draw_labeled_bboxes(img, labels)
        #plt.imshow(draw_img)
        #plt.show()

        bbox = self.ipu.get_labeled_bboxes(labels)

        out_img = draw_img
        return out_img, bbox

    # Main pipeline process of this project
    def pipeline_yolov2(self, img):

        # apply YOLOv2
        out_img, pred_boxes, pred_classes, pred_scores = self.annotate.make_annotate(img, threshold=0.5)

        max_s = float("-inf")
        bbox = []
        for b, c, s in zip(pred_boxes, pred_classes, pred_scores):
            if c == self.target:
                if s > max_s:
                    bbox = [[[b[1], b[0]], [b[3], b[2]]]]
                    max_s = s

        return out_img, bbox

    # run the main pipeline for video
    def run_pipeline(self, video_file, duration=None, end=False):
        """Runs pipeline on a video and writes it to temp folder"""
        print('processing video file {}'.format(video_file))
        clip = VideoFileClip(video_file)

        if duration is not None:
            if end:
                clip = clip.subclip(clip.duration - duration)
            else:
                clip = clip.subclip(0, duration)

        fpath = 'temp/' + video_file
        if os.path.exists(fpath):
            os.remove(fpath)
        if self.do_svc:
            processed = clip.fl(lambda gf, t: self.pipeline_svc(gf(t)), [])
        if self.do_kb:
            processed = clip.fl(lambda gf, t: self.pipeline_kb(gf(t)), [])
        processed.write_videofile(fpath, audio=False)


def estimate_obstacle_car(bbox):
    bbox = bbox[0]
    cx = abs((bbox[1][0] + bbox[0][0]) / 2)
    cy = abs((bbox[1][1] + bbox[0][1]) / 2)

    dx = abs((bbox[1][0] - bbox[0][0]))
    dy = abs((bbox[1][1] - bbox[0][1]))

    # tx:dy = 28.7 : 75.9 = 4 : 320
    a = (4 - 28.7) / (320 - 75.9)
    a += -a / 1.5
    b = 28.7 - a * 75.9 - 12 # + is far, - is closer
    tx = a * dy + b
    #tx = -0.1 * dy + 36.0

    # ty:cx = 0.2 : 680 = 3.6 : 1293
    a = (0.2 - 3.6) / (680. - 1293)
    a += a / 3.7
    b = 3.6 - a * 680 # + is right, - is left
    ty = a * cx + b
    ty = -ty + 3.0
    tz = -0.85
    return tx, ty, tz

def estimate_obstacle_ped(bbox):
    bbox = bbox[0]
    cx = abs((bbox[1][0] + bbox[0][0]) / 2)
    cy = abs((bbox[1][1] + bbox[0][1]) / 2)

    dx = abs((bbox[1][0] - bbox[0][0]))
    dy = abs((bbox[1][1] - bbox[0][1]))

    # tx:dy = 8.3 : 7.5 = 340 : 360
    a = (8.3 - 7.5) / (340 - 360)
    a = a / 3.5
    b = 8.3 - a * 340 # + is far, - is closer
    tx = a * dy + b

    # ty:cx = 0.2 : 680 = 3.6 : 1293
    a = (0.2 - 3.6) / (680. - 1293)
    a += a / 3.7
    b = 3.6 - a * 680 # + is right, - is left
    ty = a * cx + b
    ty = -ty + 3.0
    tz = -0.85
    return tx, ty, tz

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def main():

    do_images = True
    do_csv = False

    if do_images:
        pl = Pipeline()

    # for dataset of round 1
    #with open("test_images/metadata.csv", "r") as f:
    #    meta_data = f.readlines()
    #l = float(str.split(meta_data[1], ",")[2])
    #w = float(str.split(meta_data[1], ",")[3])
    #h = float(str.split(meta_data[1], ",")[4])

    #dir_names = ["ford01", "ford02", "ford03", "ford04", "ford05", "ford06", "ford07", "mustang01", "ped_test"]
    #dir_names = ["ford07", "mustang01", "ped_test"]
    dir_names = ["ford01", "ford02", "ford03"]

    for dir_name in dir_names:
        if dir_name == "ped_test":
            target = "person"
            l = float(0.8)
            w = float(0.8)
            h = float(1.7)
        else:
            target = "car"
            l = float(4.0)
            w = float(1.7)
            h = float(1.5)

        # prepare to generate tracklets
        path = os.path.join("outputs", dir_name, "camera")
        timestamps = 0
        collection = TrackletCollection()
        obs_tracklet = Tracklet(object_type=target, l=l, w=w, h=h, first_frame=timestamps)
        tracklet_file = os.path.join("outputs", "submit", dir_name + '.xml')

        # pipelines
        if do_images:
            images = sorted(glob.glob(os.path.join(path, '*.jpg')), key=numericalSort)
            lines = []
            pl.target = target

            for fname in images:
                print("{}/{}".format(timestamps, len(images)))

                if pl.do_kb:
                    image = mpimg.imread(fname)
                    output, bbox = pl.pipeline_kb(image)
                    mpimg.imsave(os.path.join('output_images', dir_name, 'output_' + os.path.splitext(os.path.basename(fname))[0] + ".png"), output)
                if pl.do_yolov2:
                    image = Image.open(fname)
                    image = image.crop((0, 0, image.width, image.height - 250))
                    output, bbox = pl.pipeline_yolov2(image)
                    resized_image = image.resize((int(image.width/3), int(image.height/3)))
                    mpimg.imsave(os.path.join('output_images', dir_name, 'output_' + os.path.splitext(os.path.basename(fname))[0] + ".png"), resized_image)

                line = os.path.basename(fname)

                if len(bbox) > 0:
                    for item in bbox[0]:
                        line += ", " + str(item[0])
                        line += ", " + str(item[1])
                else:
                    line += ",0,0,0,0"
                lines.append(line)
                #plt.imshow(output)
                pl.reset()

                if len(bbox) > 0:
                    if dir_name == "ped_test":
                        tx, ty, tz = estimate_obstacle_ped(bbox)
                    else:
                        tx, ty, tz = estimate_obstacle_car(bbox)
                else:
                    tx = 0
                    ty = 0
                    tz = 0

                obs_tracklet.poses.append({'tx': tx, 'ty': ty, 'tz': tz, 'rx': 0, 'ry': 0, 'rz': 0, })
                timestamps += 1
            collection.tracklets = [obs_tracklet]

            ## save
            collection.write_xml(tracklet_file)
            with open(os.path.join(path, 'box.csv'), 'w', newline='\n') as csvfile:
                for line in lines:
                    csvfile.writelines(line + "\n")

        if do_csv:
            file_path = os.path.join(path, 'box.csv')
            length = len(open(file_path).readlines())

            with open(file_path, 'r') as csvfile:
                import csv
                lines = csv.reader(csvfile)

                for line in lines:
                    x1, y1, x2, y2 = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    bbox = [[[x1, y1], [x2, y2]]]
                    if x1 == 0. and y1 == 0. and x2 == 0. and y2 == 0.:
                        tx, ty, tz = 0., 0., 0.
                    else:
                        if dir_name == "ped_test":
                            tx, ty, tz = estimate_obstacle_ped(bbox)
                        else:
                            tx, ty, tz = estimate_obstacle_car(bbox)
                    print("{}/{}:{}".format(timestamps, length, bbox))
                    obs_tracklet.poses.append({'tx': tx, 'ty': ty, 'tz': tz, 'rx': 0, 'ry': 0, 'rz': 0, })
                    timestamps += 1

            collection.tracklets = [obs_tracklet]
            ## save
            collection.write_xml(tracklet_file)
            collection.write_xml("/home/keigo/catkin_ws/src/data/" + dir_name + '.xml')




if __name__ == '__main__':
    main()
