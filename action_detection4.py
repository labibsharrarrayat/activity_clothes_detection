from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
from keras import backend as K
K.set_image_dim_ordering('th')

import argparse
import logging
import time
from darkflow.net.build import TFNet

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

options = {'model': 'cfg/tiny-yolo-voc.cfg','load': 'bin/tiny-yolo-voc.weights','threshold': 0.4,'gpu': 0.7}
options2 = {'model': 'cfg/tiny-yolo-voc-2c.cfg','load':7315,'threshold': 0.2,'gpu': 0.7}

tfnet = TFNet(options)
tfnet2 = TFNet(options2)


colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


model = load_model("activity_new_version.h5")



fps_time = 0

count = 0

def action_detection(img):

    img = cv2.resize(img, (150, 150))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    pred = pred[0]

    action = int(pred)

    if action == 1:
        act_label = "standing"

    elif action == 0:
        act_label = "sitting"

    return act_label



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()


    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)


    ret_val, image = cam.read()


    while True:
        ret_val, image = cam.read()
        frame = image

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        image = np.zeros((480, 630, 3), np.uint8)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        time_diff = time.time() - fps_time

        cv2.putText(image,
                    "FPS: %f" % (1.0 / time_diff),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        if ret_val:
            results = tfnet.return_predict(frame)
            result2s = tfnet2.return_predict(frame)

            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)

                if(label == 'person'):


                    x = tl[0]
                    y = tl[1]

                    x2 = br[0] - tl[0]
                    y2 = br[1] - tl[1]

                    w1 = x + x2
                    h1 = y + y2

                    sub_img = image[y:h1, x:w1]

                    action_label = action_detection(sub_img)


                    frame = cv2.rectangle(frame, (x, y), (w1, h1), (0, 0, 255), 2)
                    frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 69, 255), 2)
                    frame = cv2.putText(frame,action_label,(tl[0],(tl[1]-30)),cv2.FONT_HERSHEY_COMPLEX,1,(80,127,255),2)



                    for color, result2 in zip(colors, result2s):
                        tl2 = (result2['topleft']['x'], result2['topleft']['y'])
                        br2 = (result2['bottomright']['x'], result2['bottomright']['y'])
                        label2 = result2['label']
                        confidence2 = result2['confidence']
                        text2 = '{}'.format(label2)

                        frame = cv2.putText(frame, text2,(tl[0], (tl[1] - 60)), cv2.FONT_HERSHEY_COMPLEX, 1,(71, 99, 255), 2)



            cv2.imshow('tf-pose-estimation result', image)
            cv2.imshow('frame',frame)
            fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break


cv2.destroyAllWindows()





