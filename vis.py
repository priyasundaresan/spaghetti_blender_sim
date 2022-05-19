import cv2
import numpy as np
import os

def overlay_pusher(image, annots):
    pusher_pixel = annots[0]
    hull_pixels = annots[1:-1]
    densest_pixel = annots[-1]
    cv2.circle(image, tuple(pusher_pixel), 3, (255,255,0), -1)
    cv2.circle(image, tuple(densest_pixel), 3, (255,0,0), -1)
    for px in hull_pixels:
        cv2.circle(image, tuple(px), 3, (0,0,255), -1)
    return img
    #cv2.imshow('img', image)
    #cv2.waitKey(0)

if __name__ == '__main__':
    annots_dir = 'annots'
    img_dir = 'masks'
    vis_dir = 'vis'
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    for i, img_fn in enumerate(sorted(os.listdir(img_dir))):
        img = cv2.imread(os.path.join(img_dir, img_fn))
        annot = np.load(os.path.join(annots_dir, '%03d.npy'%i))[0]
        vis = overlay_pusher(img, annot)
        cv2.imwrite('%s/%03d.jpg'%(vis_dir,i), vis)
        
    
