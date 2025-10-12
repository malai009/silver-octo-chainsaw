import cv2
import os
import mediapipe as mp
import argparse

def process_img(img, face_detection):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we want img to be in rgb for face detection
    img_out = face_detection.process(img_rgb) # output img after processing
    
    H, W, _ = img.shape

    if img_out and img_out.detections is not None:  # to make sure the loop runs iff the image contains a human face, not an animal face

      for detection in img_out.detections: #these data have been copy pasted from the output of print from termial
        location_data = detection.location_data
        bbx = location_data.relative_bounding_box

        x1, y1, w, h = bbx.xmin, bbx.ymin, bbx.width, bbx.height
        
        X1 = int(x1 * W)   # converting the coordinates of bbx into absolute values as they were previously 
        Y1 = int(y1 * H)   # relative
        W = int(w * W)
        H = int(h * H)

        ## Blur Faces
        img_blur = cv2.blur(img[Y1:Y1+H, X1:X1+W, :], (80,80))
        img[Y1:Y1+H, X1:X1+W, :] = img_blur
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode", default ='video')    #defining if the data to be taken from an image or video or webcam
args.add_argument("--filepath", default = 'face_video.mp4')

args = args.parse_args()

## Detect Faces using Mediapie library
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection( model_selection = 0, min_detection_confidence = 0.5) as face_detection:
   
    if args.mode in ['image']:
      ## Read image
        img_path = 'face_david.webp'

        img = cv2.imread(args.filepath)

        img = process_img(img, face_detection)

        ## Save Image
        cv2.imwrite(os.path.join('.', 'faceBlurOP.png'),img)

    elif args.mode in ['video']:
      cap = cv2.VideoCapture(args.filepath)
      ret, frame = cap.read()

      output_video = cv2.VideoWriter(os.path.join('.', 'output.mp4'),
                                     cv2.VideoWriter.fourcc(*'MP4V'),
                                     25, (frame.shape[1], frame.shape[0]))


      while ret:
         frame = process_img(frame, face_detection)

         output_video.write(frame)

         ret, frame = cap.read()

      cap.release()
      output_video.release()


#print(img_out.detections) # this is done to get the bounding box location of info of face
#cv2.rectangle(img, (X1,Y1), (X1+W,Y1+H), [0, 255, 0], 1 )

#cv2.imshow('img', img)
#cv2.waitKey(0)

# making it work with a video

