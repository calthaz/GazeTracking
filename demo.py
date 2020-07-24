#!/usr/bin/env python3


"""
Demonstration of how to use the eye point of gaze (EPOG) tracking library.

This example application can be called like this (both args are optional):
>> ./epog_example.py 1 'log_file_prefix'

'1': stabilize estimated EPOG w.r.t. previous cluster of EPOGs
'0': allow spurious EPOGs that deviate from cluster (default)

'log_file_prefix': (e.g. user_id) A logfile will be created with the errors, i.e.
the Euclidean distance (in pixels) between test points and corresponding estimated EPOGs.
Log file will be e.g. test_errors/'log_file_prefix'_stab_01-12-2019_18.36.44.txt
If log_file_prefix is omitted, log file will not be created.

Check the README.md for complete documentation.
"""

import sys
import cv2
import gaze_tracking as gt
import numpy as np

def not_detected_frame(w, h):
    fullscreen_frame = np.zeros((h, w, 3), np.uint8)
    cv2.putText(fullscreen_frame, "Not Detected", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
    return fullscreen_frame

def put_pic_center(frame, pic, box):
    pic_width = pic.shape[1]
    pic_height = pic.shape[0]
    #print(pic_width, pic_height)
    box_width = box[1][0]- box[0][0]
    box_height = box[1][1]- box[0][1]
    #print(box_width, box_height)
    ratio = min(box_width/pic_width, box_height/pic_height)
    #print(ratio)
    resized_pic = cv2.resize(pic, None, fx = ratio, fy=ratio)
    resized_width = resized_pic.shape[1]
    resized_height = resized_pic.shape[0]
    #print(resized_width, resized_height)
    margin_left = 0
    margin_top = 0
    if(resized_width<box_width):
        margin_left = (box_width-resized_width)//2
    if(resized_height<box_height):
        margin_top = (box_height-resized_height)//2
    frame[box[0][1]+margin_top: box[0][1]+margin_top+resized_height, box[0][0]+margin_left: box[0][0]+margin_left+resized_width] = resized_pic
    return frame



# setup_epog expects max two args, both optional,
# sets up webcam, and calibration windows
test_error_dir = '../GazeEvaluation/test_errors/'
epog = gt.EPOG(test_error_dir, sys.argv)

monitor = gt.get_screensize()  # dict: {width, height}
w = monitor["width"]
h = monitor['height']
start_conference = False

alice_c = cv2.imread('img/Alice_c.jpg')
alice_nc = cv2.imread("img/Alice_nc.jpg")
me_c = cv2.imread('img/Me_c.jpg')
me_nc = cv2.imread("img/Me_nc.jpg")
bob_c = cv2.imread('img/Bob_c.jpg')
bob_nc = cv2.imread("img/Bob_nc.jpg")

while True:
    # We get a new frame from the webcam
    _, frame = epog.webcam.read()
    if frame is not None:
        # Analyze gaze direction and map to screen coordinates
        screen_x, screen_y = epog.analyze(frame)

        # Access gaze direction
        text = ""
        if epog.gaze_tr.is_right():
            text = "Looking right"
        elif epog.gaze_tr.is_left():
            text = "Looking left"
        elif epog.gaze_tr.is_center():
            text = "Looking center"

        focused_person = None #0 alice, 1 bob
        
        # Use gaze projected onto screen surface
        # Screen coords will be None for a few initial frames,
        # before calibration and tests have been completed
        if screen_x is not None and screen_y is not None:
            text = "Looking at point {}, {} on the screen".format(screen_x, screen_y)
            start_conference = True
            if screen_x < w*3/10:
                if screen_y < h*5/9:
                    focused_person = "Alice"
                else:
                    focused_person = "Bob"

        if start_conference:    
            fullscreen_frame = np.zeros((h, w, 3), np.uint8)
            flipped_me = cv2.flip(frame, 1)#flip around y axis
            
            
            my_alice_box = [[0, h//9], [w*3//10, h*5//9]]
            fullscreen_frame=put_pic_center(fullscreen_frame, alice_c, my_alice_box)

            my_bob_box = [[0, h*5//9], [w*3//10, h]]
            fullscreen_frame=put_pic_center(fullscreen_frame, bob_c, my_bob_box)
            
            my_box = [[w*3//10, h//9], [w*6//10, h]]
            fullscreen_frame=put_pic_center(fullscreen_frame, flipped_me, my_box)
            
            alice_alice_box = [[w*4//5, h//10], [w, h//2]]
            fullscreen_frame=put_pic_center(fullscreen_frame, alice_c, alice_alice_box)
            
            alice_bob_box = [[w*3//5, h*3//10], [w*4//5, h//2]]
            fullscreen_frame=put_pic_center(fullscreen_frame, bob_nc, alice_bob_box)

            bob_alice_box = [[w*3//5, h*6//10], [w*4//5, h*8//10]]
            fullscreen_frame=put_pic_center(fullscreen_frame, alice_nc, bob_alice_box)
            
            bob_bob_box = [[w*4//5, h*6//10], [w, h]]
            fullscreen_frame=put_pic_center(fullscreen_frame, bob_c, bob_bob_box)
            
            alice_me_box = [[w*3//5, h*1//10], [w*4//5, h*3//10]]
            bob_me_box = [[w*3//5, h*8//10], [w*4//5, h]]
            if(focused_person=="Alice"):
                fullscreen_frame = put_pic_center(fullscreen_frame, me_c, alice_me_box)
                fullscreen_frame = put_pic_center(fullscreen_frame, me_nc, bob_me_box)
                cv2.rectangle(fullscreen_frame, (my_alice_box[0][0], my_alice_box[0][1]), 
                (my_alice_box[1][0], my_alice_box[1][1]), (147, 58, 31), 5) 
                cv2.rectangle(fullscreen_frame, (alice_me_box[0][0], alice_me_box[0][1]), 
                (alice_me_box[1][0], alice_me_box[1][1]), (147, 58, 31), 5) 
            elif (focused_person=="Bob"):
                fullscreen_frame = put_pic_center(fullscreen_frame, me_nc, alice_me_box)
                fullscreen_frame = put_pic_center(fullscreen_frame, me_c, bob_me_box)
                cv2.rectangle(fullscreen_frame, (my_bob_box[0][0], my_bob_box[0][1]), 
                (my_bob_box[1][0], my_bob_box[1][1]), (147, 58, 31), 5) 
                cv2.rectangle(fullscreen_frame, (bob_me_box[0][0], bob_me_box[0][1]), 
                (bob_me_box[1][0], bob_me_box[1][1]), (147, 58, 31), 5) 
            else:
                fullscreen_frame = put_pic_center(fullscreen_frame, me_nc, alice_me_box)
                fullscreen_frame = put_pic_center(fullscreen_frame, me_nc, bob_me_box)
            

            cv2.putText(fullscreen_frame, "Me", (60, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
            cv2.putText(fullscreen_frame, "Alice", (60+w*3//5, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
            cv2.putText(fullscreen_frame, "Bob", (60+w*3//5, 50+h//2), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
           
            cv2.namedWindow("Me", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Me",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            #cv2.putText(frame, "here", (screen_x, screen_y), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
            #fullscreen_frame = np.transpose(fullscreen_frame, (1, 0, 2))
            cv2.imshow("Me", fullscreen_frame)


        # Press Esc to quit the video analysis loop
        if cv2.waitKey(1) == 27:
            # Release video capture
            epog.webcam.release()
            cv2.destroyAllWindows()
            break
        # Note: The waitkey function is the only method in HighGUI that can fetch and handle events,
        # so it needs to be called periodically for normal event processing unless HighGUI
        # is used within an environment that takes care of event processing.
        # Note: The waitkey function only works if there is at least one HighGUI window created and
        # the window is active. If there are several HighGUI windows, any of them can be active.
        # (https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html)
