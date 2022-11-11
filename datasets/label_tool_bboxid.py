import numpy as np
import cv2
import sys
import os
import pickle
import pandas as pd

video_name = sys.argv[1]
if 'am_singles' in video_name:
    bboxid_root = '../data_annotations/object_level/bbox_id_am_singles/' 
    list_ids = [1,2]
elif 'pro' in video_name:
    bboxid_root = '../data_annotations/object_level/bbox_id_pro/'
    list_ids = [1,2]
elif 'am_doubles' in video_name:
    bboxid_root = '../data_annotations/object_level/bbox_id_am_doubles/'
    list_ids = [1,2,3,4]
else:
    print('bboxid directory not found, should only have bbox_id_am_singles/,bbox_id_am_doubles/, bbox_id_pro/')

root_dir =  os.path.join('/'.join(video_name.split('/')[:-3]))
match_dir = video_name.split('/')[-3]
gt_bboxid_dir = os.path.join(root_dir, match_dir, 'bbox_id')

if not os.path.exists(gt_bboxid_dir):
    os.makedirs(gt_bboxid_dir)

basename = video_name.split('/')[-1][:-4]
filename = os.path.join(gt_bboxid_dir, basename)
outputfile_name1 = filename + '_gtbboxid.csv'
pred_bboxid_file = os.path.join(bboxid_root, match_dir, basename) + '_id_pose.csv'
print(pred_bboxid_file)

# name="WS_TAITzuYing_vs_CHENYuFei"
# ext=".mp4"
# filename=name+ext
#filename = video_name.split(os.sep)[-1].split('.')[0]
data=dict()
racket = dict()
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global data,cap,current, current_id
    global image
    
    # if the left mouse button was clicked, record the starting (x, y) coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        if current not in data:
            data[current] = dict()
        data[current][current_id] = [(x,y)]
    # check to see if left mouse button was eleased
    elif event == cv2.EVENT_LBUTTONUP:
        # record enging (x,y) coords
        data[current][current_id].append((x,y))
        image=toframe(cap,current,total_frame)

def toframe(cap,n,total_frame):
    # print('current frame: ',n)
    cap.set(cv2.CAP_PROP_POS_FRAMES,n); 
    ret, frame = cap.read()
    
    if not ret:
        return None
    else:
        if current in data:
            for id in list_ids:
                if id in data[current] and len(data[current][id])==2:
                    cv2.rectangle(frame, data[current][id][0], data[current][id][1], (0, 0, 255), 2)
                    cv2.putText(frame,str(id),data[current][id][0], cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

total_frame=0
cap = cv2.VideoCapture(video_name)
total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
print ("Total frame : "+str(total_frame))
full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
full_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
width, height = full_width, full_height # resize to smaller frame so faster to annotate

try :
    csv_data = pd.read_csv(pred_bboxid_file)

    csv_frame = csv_data['Frame'].values
    csv_id = csv_data['id'].astype(int).values
    csv_x1 = csv_data['x1'].values
    csv_y1 = csv_data['y1'].values
    csv_x2 = csv_data['x2'].values
    csv_y2 = csv_data['y2'].values

    # check if coords are in pixels or ratio form
    if np.count_nonzero(csv_x1<1) > 0.5*len(csv_x1):
        csv_x1, csv_y1, csv_x2,csv_y2 = csv_x1*width, csv_y1*height, csv_x2*width,csv_y2*height
    else:
        csv_x1, csv_y1, csv_x2,csv_y2 = csv_x1*width/full_width, csv_y1*height/full_height, csv_x2*width/full_width,csv_y2*height/full_height

    for i in range(len(csv_frame)):
        curr_frame = csv_frame[i]
        curr_id = csv_id[i]
        if curr_frame not in data:
            data[curr_frame] = dict()
        data[curr_frame][curr_id] = [(int(csv_x1[i]), int(csv_y1[i])), (int(csv_x2[i]), int(csv_y2[i]))]
    
except Exception as e:
    print ('\nThis video has not been predicted! Good Luck!!')

current=0
current_id = 1 # set by default
image=toframe(cap,current,total_frame)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
saved=False

try:
    data,racket=pickle.load(open(filename+".pkl",'rb'))
    print ("loaded from "+filename+".pkl")
    if max(data.keys()) > max(racket.keys()):
        print ("min frame ", str(min(data.keys())))
        print ("max frame ", str(max(data.keys())))
        print ("jump to max frame")
        current=max(data.keys())
    else:
        print ("min frame ", str(min(racket.keys())))
        print ("max frame ", str(max(racket.keys())))
        print ("jump to max frame")
        current=max(racket.keys())
    image=toframe(cap,current,total_frame)
except Exception as e:
    print ('\nThis is new video! Good Luck!!')
# keep looping until the 'q' key is pressed

while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("w"):		# delete ball point
        if current in data:
            del data[current]
            print('\nYou delete the ball coordinate.')
            image=toframe(cap,current,total_frame)
        else:
            print('\nNo ball coordinate!!')
    elif key == ord("a"):		# delete ball point
        if current in data:
            if current_id in data[current]:
                del data[current][current_id]
                print('\nYou delete the ball coordinate.')
                image=toframe(cap,current,total_frame)
        else:
            print('\nNo ball coordinate!!')
    elif key == ord("1"):
        current_id = 1
        image=toframe(cap,current,total_frame)
    elif key == ord("2"):
        current_id = 2
        image=toframe(cap,current,total_frame)
    elif key == ord("3"):
        current_id = 3
        image=toframe(cap,current,total_frame)
    elif key == ord("4"):
        current_id = 4
        image=toframe(cap,current,total_frame)
    elif key == ord("f"):
        print(current)
        current=int(input('Enter your frame:'))
        image=toframe(cap,current,total_frame)
    elif key == ord("n"):     #jump next 30 frames
        check = current+30
        if check < total_frame-1:
            current+=30
            
        else:
            current = total_frame-1
            print('\nThis is last frame.')
        image=toframe(cap,current,total_frame)
    elif key == ord("p"):     #jump last 30 frames
        check = current-30
        if check <= 0:
            print('\nInvaild !!! Jump to first image...')
            current = 0
        else:
            current = check
        image=toframe(cap,current,total_frame)
    elif key == ord("d"):     #jump next frame
        if current < total_frame-1:
            current+=1
            image=toframe(cap,current,total_frame)
        else:
            print('\nCongrats! This is the last frame!!')
    elif key == ord("e"):     #jump last frame
        if current == 0:
            print('\nThis is first images')
        else:
            current-=1
        image=toframe(cap,current,total_frame)
    elif key == ord("s"):     #save as .pkl
        saved = True
        try:
            pickle.dump([data,racket],open(filename+".pkl",'wb'))
            
            print ("saved to "+filename+".pkl")
        except Exception as e:
            print (str(e))
            
    elif key == ord("q"):
        if saved:
            break
        else:
            print('You DONT save the data!!')

cv2.destroyAllWindows()
cap.release()

row_list = []
for frnum in range(int(total_frame)):
    for id in list_ids:
        row_dict = {}
        if frnum in data and id in data[frnum]:
            row_dict['Frame'] = frnum
            row_dict['id'] = id
            row_dict['x1'] = data[frnum][id][0][0]/width
            row_dict['y1'] = data[frnum][id][0][1]/height
            row_dict['x2'] = data[frnum][id][1][0]/width
            row_dict['y2'] = data[frnum][id][1][1]/height
            row_dict['Visbility'] = 1
        else:
            row_dict['Frame'] = frnum
            row_dict['id'] = id
            row_dict['x1'] = 0
            row_dict['y1'] = 0
            row_dict['x2'] = 0
            row_dict['y2'] = 0
            row_dict['Visbility'] = 0
        row_list.append(row_dict)

df_labels = pd.DataFrame(row_list)
df_labels.to_csv(outputfile_name1, encoding='utf-8', index=False)

