# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:32:32 2019

@author: Mayank
"""

import cv2
import numpy as np
import math
import serial
import time

"""
9600- Baud Rate
COM11- Port Number
"""

ser = serial.Serial('COM11',9600)   # for serial communication

"""
0- Internal (default) Webcam
1- External Webcam
cv2.VideoCapture(<path>): This path is the video path.
Basically 1 and 0 are like Path of the video.
"""

cap = cv2.VideoCapture(1)

"""
For setting size of the video.
"""
cap.set(3,960)
cap.set(4,1280)

n=9   # size of matrix

#bfs_function
def bfs(root,goal):
    queue=[]
    visited=[]
    queue.append(root)              #
    visited.append(root)
    parent=np.zeros(n*n,int)
    parent=parent-1
    while queue:
        vertex=queue[0]
        childlist=[]
        queue.pop(0)

        for i in range(n*n):
            if adj[vertex,i]==1:
                childlist.append(i)
        for child in childlist:
            if child not in visited:
                visited.append(child)
                parent[child]=vertex
                child_i=child//n
                child_j=child%n
                if p_array[child_i,child_j]==goal:
                    return parent,visited
                else:
                    queue.append(child)
    return 0


# returns center of pink, brown co-ordinates
def bot_pos():
    _, frame = cap.read()
    frame=frame[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]
    maskbr=cv2.inRange(frame,lower_brown-20,upper_brown+20)
    maskp=cv2.inRange(frame,lower_pink-20,upper_pink+20)
    
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(maskbr,kernel,iterations = 1)
    maskbrn = cv2.dilate(erosion,kernel,iterations = 1)

    erosion = cv2.erode(maskp,kernel,iterations = 1)
    maskpn = cv2.dilate(erosion,kernel,iterations = 1)
    
    
    
    contours_brown,hierachy =cv2.findContours(maskbrn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    r_areas = [cv2.contourArea(c) for c in contours_brown]

    max_rarea = np.max(r_areas)
            
    for cnt in contours_brown:
        approx=cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
        if(( cv2.contourArea(cnt) > max_rarea * 0.50) and (cv2.contourArea(cnt)<= max_rarea)):
            M = cv2.moments(approx)
            cX_brown=int(M['m10']/M['m00'])
            cY_brown=int(M['m01']/M['m00'])
        
    contours_pink,hierachy =cv2.findContours(maskpn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    r_areas = [cv2.contourArea(c) for c in contours_pink]
        
    max_rarea = np.max(r_areas)

    for cnt in contours_pink:
        approx=cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
        if(( cv2.contourArea(cnt) > max_rarea * 0.50) and (cv2.contourArea(cnt)<= max_rarea)):
            M = cv2.moments(approx)
            cX_pink=int(M['m10']/M['m00'])
            cY_pink=int(M['m01']/M['m00'])
    return np.array([cX_pink,cY_pink,cX_brown,cY_brown])


#image_input
    
"""
_,im=cap.read() === ret,frame=cap.read()
cap.read() returns 2 parameters-> ret and frame.
ret= boolean value which returns whether a frame exists or not.
frame= openCV vector of image
"""

_, im=cap.read()
#im=cv2.imread("main.jpg")

"""
CrossHair= + sign when scope is used in Snipers in Games
FromCenter= To drag the part of the image from Top Left corner to Bottom Right Corner
"""

showCrosshair = False
fromCenter = False

"""
SelectROI returns a List of 4 elements- the co-ordinates of the TOP LEFT CORNER (say x1,y1) and BOTTOM RIGHT CORNER (say x2,y2)
Hence you crop the image of the selected region
"""
r1 = cv2.selectROI("Crop arena", im, fromCenter, showCrosshair)
"""
Select the Region of Interest.
To detect the current colors
"""
im = im[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]

cv2.imwrite('main.jpg',im)

# image=1280 * 960 , image.shape=[no. of rows, no. of columns]
columns=im.shape[1]
rows=im.shape[0]


showCrosshair = False
fromCenter = False

"""
Now we will start getting the different types of colors present in the real atmosphere!
"""

#red
r = cv2.selectROI("Select red", im, fromCenter, showCrosshair)
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

"""
Select Lower and Upper Range Values of the colors to make a mask
"""
lower_red=np.array([imCrop[:,:,0].min(),imCrop[:,:,1].min(),imCrop[:,:,2].min()])
upper_red=np.array([imCrop[:,:,0].max(),imCrop[:,:,1].max(),imCrop[:,:,2].max()])

"""
-35 and +35 are done to get range of colors because here we have taken only one ROI
"""
maskr=cv2.inRange(im,lower_red-35,upper_red+35) #Masking

"""
Filter and Dilate the image for good corners
"""
kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(maskr,kernel,iterations = 1)
maskrn = cv2.dilate(erosion,kernel,iterations = 1)

cv2.imshow('red',maskrn)
cv2.waitKey(0)



#yellow
r = cv2.selectROI("Select yellow", im, fromCenter, showCrosshair)
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

lower_yellow=np.array([imCrop[:,:,0].min(),imCrop[:,:,1].min(),imCrop[:,:,2].min()])
upper_yellow=np.array([imCrop[:,:,0].max(),imCrop[:,:,1].max(),imCrop[:,:,2].max()])

masky=cv2.inRange(im,lower_yellow-20,upper_yellow+20)

kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(masky,kernel,iterations = 1)
maskyn = cv2.dilate(erosion,kernel,iterations = 1)

cv2.imshow('yellow',maskyn)
cv2.waitKey(0)


#brown
r = cv2.selectROI("Select brown", im, fromCenter, showCrosshair)
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

lower_brown=np.array([imCrop[:,:,0].min(),imCrop[:,:,1].min(),imCrop[:,:,2].min()])
upper_brown=np.array([imCrop[:,:,0].max(),imCrop[:,:,1].max(),imCrop[:,:,2].max()])

maskbr=cv2.inRange(im,lower_brown-10,upper_brown+10)

kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(maskbr,kernel,iterations = 1)
maskbrn = cv2.dilate(erosion,kernel,iterations = 1)

cv2.imshow('brn',maskbrn)
cv2.waitKey(0)


#pink
r = cv2.selectROI("Select pink", im, fromCenter, showCrosshair)
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

lower_pink=np.array([imCrop[:,:,0].min(),imCrop[:,:,1].min(),imCrop[:,:,2].min()])
upper_pink=np.array([imCrop[:,:,0].max(),imCrop[:,:,1].max(),imCrop[:,:,2].max()])

maskp=cv2.inRange(im,lower_pink-10,upper_pink+10)

kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(maskp,kernel,iterations = 1)
maskpn = cv2.dilate(erosion,kernel,iterations = 1)

cv2.imshow('pn',maskpn)
cv2.waitKey(0)


#position_array
p_array=np.zeros([n,n],int)

#contours

"""
Learn these commands especially cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
countour_red,contour_yellow = 
* Contours is a list, or tree of lists of points. The points describe each contour, 
that is, a vector that could be drawn as an outline 
around the parts of the shape based on it's difference from a background.
* contourArea= From a given List of Outline Points of contours, it finds area of each.
"""
contours_yellow,hierachy =cv2.findContours(maskyn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_red,hierachy =cv2.findContours(maskrn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#font=cv2.FONT_HERSHEY_SIMPLEX
r_areas = [cv2.contourArea(c) for c in contours_yellow]
max_rarea = np.max(r_areas)

for cnt in contours_yellow:
    approx=cv2.approxPolyDP(cnt,0.045*cv2.arcLength(cnt,True),True)
    """
    Approx Polygon Douglas-Peucker algorithm
    arcLength(curve, closed) -> Calculates a contour perimeter or a curve length. . . 
    The function computes a curve length or a closed contour perimeter. . . @param curve Input vector of 2D points, 
    stored in std::vector or Mat. . 
    @param closed Flag indicating whether the curve is closed or not.
    """
    if(( cv2.contourArea(cnt) > max_rarea * 0.02) and (cv2.contourArea(cnt)<= max_rarea)):
        cv2.drawContours(im,[approx],0,(1),4)
        M = cv2.moments(approx)
        cX=int(M['m10']/M['m00'])
        cY=int(M['m01']/M['m00'])    
        
        j=(cX*n)//columns
        i=(cY*n)//rows
        
        if(maskrn[cY,cX]==255):
            p_array[i,j]=1
        
        else:
            if len(approx)==4:
                p_array[i,j]=2
            elif len(approx)>4:
                p_array[i,j]=3

r_areas = [cv2.contourArea(c) for c in contours_red]
max_rarea = np.max(r_areas)


for cnt in contours_red:
    approx=cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
    if(( cv2.contourArea(cnt) > max_rarea * 0.20) and (cv2.contourArea(cnt)<= max_rarea)):
        cv2.drawContours(im,[approx],0,(1),4)
        M = cv2.moments(approx)
        cX=int(M['m10']/M['m00'])
        cY=int(M['m01']/M['m00'])    
        
        j=(cX*n)//columns
        i=(cY*n)//rows
        
        if len(approx)==3:
            p_array[i,j]=4 
        elif len(approx)==4:
            p_array[i,j]=5
        else:
            p_array[i,j]=6


contours_brown,hierachy =cv2.findContours(maskbrn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

r_areas = [cv2.contourArea(c) for c in contours_brown]
max_rarea = np.max(r_areas)
            
for cnt in contours_brown:
    approx=cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
    if(( cv2.contourArea(cnt) > max_rarea * 0.50) and (cv2.contourArea(cnt)<= max_rarea)):
        cv2.drawContours(im,[approx],0,(1),4)
        M = cv2.moments(approx)
        cX_brown=int(M['m10']/M['m00'])
        cY_brown=int(M['m01']/M['m00'])
        
contours_pink,hierachy =cv2.findContours(maskpn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
r_areas = [cv2.contourArea(c) for c in contours_pink]

max_rarea = np.max(r_areas)

for cnt in contours_pink:
    approx=cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
    if(( cv2.contourArea(cnt) > max_rarea * 0.50) and (cv2.contourArea(cnt)<= max_rarea)):
        cv2.drawContours(im,[approx],0,(1),4)
        M = cv2.moments(approx)
        cX_pink=int(M['m10']/M['m00'])
        cY_pink=int(M['m01']/M['m00'])

cv2.imshow('original',im)
cv2.waitKey(0)

#bot_position

Gx=(cX_brown+cX_pink)//2
Gy=(cY_brown+cY_pink)//2
   
Gj=(Gx*n)//columns
Gi=(Gy*n)//rows  

print(lower_red,upper_red,lower_yellow,upper_yellow,lower_brown,upper_brown,lower_pink,upper_pink)
print(p_array)



#nodes_array
nodes=np.zeros([n,n],int)

"""
Prints an 2D array of
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]
 [18 19 20 21 22 23 24 25 26]
 [27 28 29 30 31 32 33 34 35]
 [36 37 38 39 40 41 42 43 44]
 [45 46 47 48 49 50 51 52 53]
 [54 55 56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69 70 71]
 [72 73 74 75 76 77 78 79 80]]
"""
for i in range(n):
    for j in range(n):
        nodes[i,j]=n*i+j


#adjacency_matrix        
adj=np.zeros([n*n,n*n],int)

"""
The following set of statements connect clockwise outer-
0->1->2->3->4->5->6->7->8->17->26->35->44->53->62->71->80->79->78.......->72->63->54->45->36->27->18->9->0
"""
for i in range(n-1):
    adj[i,i+1]=1 # first row
    adj[n*(i+1)-1,n*(i+2)-1]=1 # last row
    adj[n*n-1-i,n*n-2-i]=1
    adj[n*(n-1-i),n*(n-2-i)]=1

"""
The following set of statements connect clockwise order-
20->21->22->23->24
29              33
38              42
47              51
56<-57<-58<-59<-60
"""
for i in range(4):
    adj[20+i,20+i+1]=1
    adj[24+n*i,24+n*(i+1)]=1
    adj[60-i,59-i]=1
    adj[56-n*i,56-n*(i+1)]=1

#bot_start_pos
bot_start=nodes[Gi,Gj]


#outer_to_inner
adj[13,22]=1  
adj[22,13]=1   
adj[4,13]=1
adj[13,4]=1
        
adj[42,43]=1
adj[43,44]=1
adj[44,43]=1
adj[43,42]=1
        
adj[58,67]=1
adj[67,76]=1
adj[67,58]=1
adj[76,67]=1
        
adj[36,37]=1
adj[37,38]=1
adj[38,37]=1
adj[37,36]=1


"""
Connecting and Disconnecting the graph according to the PS conditions and starting point. 
"""
if(bot_start==4):
    adj[4,13]=0
    adj[13,4]=0
    adj[22,31]=1
    adj[31,40]=1
    adj[22,23]=0
if(bot_start==36):
    adj[36,37]=0
    adj[37,36]=0
    adj[38,39]=1
    adj[39,40]=1
    adj[38,29]=0
if(bot_start==44):
    adj[44,43]=0
    adj[43,44]=0
    adj[42,41]=1
    adj[41,40]=1
    adj[42,51]=0
if(bot_start==76):
    adj[76,67]=0
    adj[67,76]=0
    adj[58,49]=1
    adj[49,40]=1
    adj[58,57]=0

    
print(nodes)

start=bot_start


while(True):
    print("Position of bot :",start)
    print("Goal: ")
    
    goal=int(input())
    
    if(goal==-1):
        break
    while(bfs(start,goal)==0):
        print("Enter Valid Input:")
        goal=int(input())
        
    #parent_array_and_visited_array
    parent,visited=bfs(start,goal)
    
    #path_array
    path=[]
    
    if(bot_start==4):
        adj[4,13]=1
        adj[4,5]=0
        
    if(bot_start==36):
        adj[36,37]=1
        adj[36,27]=0
        
    if(bot_start==44):
        adj[44,43]=1
        adj[44,53]=0
        
    if(bot_start==76):
        adj[76,67]=1
        adj[76,75]=0
    
    temp=visited[-1]
    i=temp
    
    if(temp==31 or temp==39 or temp==41 or temp==49):
        path.append(40)
        temp=40
        
    path.append(i)
    while True:
        if(parent[i]==start):
            break
        path.append(parent[i])
        i=parent[i]
        
    path.reverse()
    
    #led_on
    
    
    print("Path is:",path)
    ser.write(b'o') #led on to start moving
    for i in path:
        
        
        pos=bot_pos()
        z_bot=complex(pos[0]-pos[2], pos[1]-pos[3])
        # z_bot= Direction Complex vector of bot positin
        Gx=(pos[2]+pos[0])//2
        Gy=(pos[1]+pos[3])//2
        
        
        #next_grid_position_pixels
        x2=(i%n+0.5)*columns//n # 0.5 is done to reach center of the next cell
        y2=(i//n+0.5)*rows//n    
        # columns= Total pixels in a column so columns//n=Pixel in one cell
        
        z_des=complex(x2-Gx,y2-Gy)
        # Complex Vector to reach destination from source.
        angle=np.angle(z_bot/z_des,deg=True)
        # To calculate angle between two vectors in degrees[deg=True]
        distance=math.sqrt(((x2-Gx)*25*n/columns)**2+((y2-Gy)*25*n/rows)**2)
        # Multiplied by 25 becuase the length of each cell is 25cm X 25cm. 
        # So basically we have tried to convert pixel distance to cm.
        thresh=3
        
        while(distance>thresh):
            if(angle>=-7 and angle<=7):
                ser.write(b'f')
                time.sleep(distance/20)
                ser.write(b's')
            elif angle>7:
                ser.write(b'l')
                time.sleep(abs(angle)/110)
                ser.write(b's')
            elif angle<-7:
                ser.write(b'r')
                time.sleep(abs(angle)/110)
                ser.write(b's')
                
            pos=bot_pos()
            
            z_bot=complex(pos[0]-pos[2], pos[1]-pos[3])
            
            Gx=(pos[2]+pos[0])//2
            Gy=(pos[1]+pos[3])//2
            
            z_des=complex(x2-Gx,y2-Gy)
            
            angle=np.angle(z_bot/z_des,deg=True)

            distance=math.sqrt(((x2-Gx)*25*n/columns)**2+((y2-Gy)*25*n/rows)**2)

    pos=bot_pos()
    Gx=(pos[2]+pos[0])//2  # Gx, Gy = Bot position in Pixels
    Gy=(pos[1]+pos[3])//2    
    Gj=(Gx*n)//columns  #Gj, Gi= Bot position in 9 X 9 matrix
    Gi=(Gy*n)//rows
    if(nodes[Gi,Gj]==temp):
        print("reached")
        
        #LED_STOP_CODE
        ser.write(b'c')
        
        if(temp==40):
            #led_blink_on_home
            ser.write(b'e')
            time.sleep(6)
            print("PS COMPLETED!")
            break;
        start=temp
        
    else:
        #this is a special case to handle exceptions
        print("not reached")
        start=nodes[Gi,Gj]

cv2.destroyAllWindows()
ser.close()
cap.close()


#PROBLEMS
'''
1) openCV color thresholding
2) openCV masking according to lightning conditions
3) contour detection epsilon value in approx polygon DP
4) BOT THRESHOLDING (BASICALLY SLEEPING TIME OF THE COMMAND) OF SPEED AND ANGLE THAT HOW MUCH TO TURN AND MOVE FORWARD
5) ARDUINO MOTOR SPEED FOR PROPER FORWARD AND BACKWARD MOVEMENTS
'''

#SUMMARY
"""
1) Libraries used- OPENCV(image processing), NUMPY(matrix creation),MATH(angle and movement),SERIAL(arduino communication),TIME
2) Started serial communication and given port and baud rate 9600
3) Video capture done using cv2.videocature(1) done using external webcam
4) Resize video for clear pixels
5) Read each frame from video FPS(frame per second) using cap.read which returns 2 parameters (1st parameter tells frame detected or not,2nd parameter image array)
6) select ROI gives top left and bottom right
7) Image crop using ROI points
#What is threshold in image processing?
#Thresholding is a type of image segmentation, where we change the pixels of an image to make the image easier to analyze. In thresholding, we convert an image from color or grayscale into a binary image, i.e., one that is simply black and white

#What is meant by masking in image processing?
#Masking is an image processing method in which we define a small 'image piece' and use it to modify a larger image. Masking is the process that is underneath many types of image processing, including edge detection, motion detection, and noise reduction.

8) Detection of shape and colours in the image: a) select ROI for each colour then cropping
                                                b) then done masking
                                                c) then erosion and dilation using kernel
                                                d) finding contours for colour using cv2.findContours(maskyn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) ?
                                                e) then made area array for each contours
                                                f) then stored max area among all
                                                g) then for shape detection used approxpolyDP (approximate polygon Douglas Peucker algorithm).Here epsilon was important which tells the accuracy of result i.e. in how deep you need to mark the edges(lesser is the epsilon,sharper is the contour)
                                                h) very less pixels are removed using a if loop
                                                i) then drawn contour
                                                j) find moments(centers) of all the contours and with the help of these centers we find the color of each and every shape.
                                                k) then this is done for all colors

9)Then prepared a position array using number of distinct points in each contour
                        **Special Trick to mention yellow triangle inside red triangle
                        if(maskrn[cY,cX]==255):
                            p_array[i,j]=1

10) Bot pos     :    a)found brown center
                     b)found pink center

                     these centers will help in determining the position of the bot and the direction in which the bot is moving currently

                     c)then determined bot center which is center of brown and pink center
                     d)located position of bot in the arena which is done using *n//rows && *n//columns

11) BFS Preprocessing ie. array population and all

                     a)prepared nodes array (0-80)
                     b)prepare 81*81 adjacency matrix for bot traversal
                     c)cnnect nodes in clockwise direction 
                     d)connect nodes going to home

                     ** while traversing we need to disconnect some of the nodes

                     e) then disconnected few nodes according to bot starting point give during competition
                     f) now while(true):
                                        a)give the input of the shape seen in the dice in the goal variable
                                        b)call bfs and return parent array and visited array
                                        c)again connecting and disconnecting some nodes
                                        d)take a temp variable to store last visited node which helps in telling when reached the destination
                                        f)with the help of parent array populate a path array which tells entire path(which could have been easily done during bfs)

12) Physical movement of the bot :
                                 a) traverse along each element in path array and we will traverse one by one from one node to next node rather than going directly to the detination in on go.
                                 b) Again get the position of bot by calling bot_pos
                                 c) now make a direction vector from brown to pink which helps the bot to determine the direction in which it should move
                                 d) Again get the center of the bot
                                 e) find center of the next cell in pixel which helps in determinig the direction in which bot has to move
                                 f) make a vector using bot center and the center of the next cell where bot has to move in pixels
                                 g) determine the angle in which the bot needs to move with the help of 

                                                    z_des=complex(x2-Gx,y2-Gy) #complex vector to reach detination
        
                                                    angle=np.angle(z_bot/z_des,deg=True)
                                h) calculate the distance per pixel keeping in mind that each cell is of 25*25...(here we are converting pixel distance to cm)
                                i) accordingly we will set the sleep time
                                j)   
                                       while(distance>thresh):
                                            if(angle>=-7 and angle<=7):
                                                ser.write(b'f')
                                                time.sleep(distance/20)# these 20,110,110 are set using motor speed practically
                                                ser.write(b's')
                                            elif angle>7:
                                                ser.write(b'l')
                                                time.sleep(abs(angle)/110)
                                                ser.write(b's')
                                            elif angle<-7:
                                                ser.write(b'r')
                                                time.sleep(abs(angle)/110)
                                                ser.write(b's')
                                                
                                            pos=bot_pos()
                                            
                                            z_bot=complex(pos[0]-pos[2], pos[1]-pos[3])
                                            
                                            Gx=(pos[2]+pos[0])//2
                                            Gy=(pos[1]+pos[3])//2
                                            
                                            z_des=complex(x2-Gx,y2-Gy)
                                            
                                            angle=np.angle(z_bot/z_des,deg=True)

                                            distance=math.sqrt(((x2-Gx)25*n/columns)2+((y2-Gy)*25*n/rows)*2)

                                i) Repeat the above mentioned steps till we did not reach the destination


xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    FINALLY PS COMPLETED    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
