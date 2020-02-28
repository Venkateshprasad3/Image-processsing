from __future__ import print_function

from os.path import join

import cv2
import imreg as imreg
import numpy as np
import os
import array as arr

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


#images=['offset_scale.jpg','offset_scale10_rot45.jpg','ofset_down_scale.jpg','rot25_offset_scale10.jpg','rot180_scale20.jpg','rot180_trans.jpg','rot300.jpg','rotation10_transformation.jpg','scale10_rot45.jpg','trans_offset_scale.jpg']
#aligned=['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg']
def alignImages(im1, im2,i):

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)


    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)


    matches.sort(key=lambda x: x.distance, reverse=False)


    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    path='matches/'
    out="match"+str(i)
    out=out+".jpg"
    print("saving match image",out)
    cv2.imwrite(os.path.join(path,out),imMatches)


    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt


    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)


    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':

        i=1
        refFilename = "original/original.jpg"
        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        rootdir="images/"
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                frame = cv2.imread(os.path.join(subdir, file))
                print("Aligning images ..."+str(i))
                imReg, h = alignImages(frame, imReference,i)
                path='output/'
                outputfilename="output"+str(i)
                outputfilename=outputfilename+".jpg"
                print("Saving aligned image : ", outputfilename)
                cv2.imwrite(os.path.join

                            (path,outputfilename),imReg)
                i=i+1
