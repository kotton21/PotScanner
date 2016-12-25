import Image
import colorsys

fn='/sdcard/bluetooth/0.jpg'
i=Image.open(fn)
d=i.getdata()

b=i.getbbox()
w=b[2]-b[0]
h=b[3]-b[1]

def hsv(rgb):
    return colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

r=h/2
blue=[]
for j in range(w):
    pi=r*h+j    #index
    p=d[pi]    #pixel value
    p_hsv = hsv(p)
    if (p[2] > 100 and p[0]<100 and p[1]<100):
        print pi,p
        blue.append(pi)
        print "hsv", hsv(p)
        
#bad. rrconvert to hsv first. need opencv?
#then set b and w thresholds on the row you want
#set to 0 if not blue?
#record min and max pixel width where within threshold
  
   
print len(blue)
print w


#loop over each row number

#def GetEdgePixel(frame, rowNumber)

#def GetThetaC(pixelCenter, pixelDensity, pixelEdge, focalDist, ...):

#def 