#!/usr/bin/env python

import os
import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def readPoints(path):
    pointsArray = []
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".txt"):
            print(f"Reading points file: {filePath}")  # Debug
            points = []
            with open(os.path.join(path, filePath)) as file:
                for line in file:
                    x, y = line.split()
                    points.append((int(x), int(y)))
            pointsArray.append(points)
    return pointsArray

def readImages(path):
    imagesArray = []
    for filePath in sorted(os.listdir(path)):
        if filePath.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Reading image file: {filePath}")  # Debug
            img = cv2.imread(os.path.join(path, filePath))
            if img is None:
                print(f"Warning: Unable to read image {filePath}")
                continue
            img = np.float32(img) / 255.0
            imagesArray.append(img)
    return imagesArray

def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]
    inPts.append([int(xin), int(yin)])
    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]
    outPts.append([int(xout), int(yout)])
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    return tform[0]

def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    for t in triangleList:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if rectContains(rect, pt[0]) and rectContains(rect, pt[1]) and rectContains(rect, pt[2]):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri

def constrainPoint(p, w, h):
    return min(max(int(p[0]), 0), w - 1), min(max(int(p[1]), 0), h - 1)

def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warpTriangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1Rect = []
    t2Rect = []
    t2RectInt = []
    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1.0 - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

def process_images(path):
    w = 600
    h = 600
    allPoints = readPoints(path)
    images = readImages(path)
    if len(allPoints) == 0 or len(images) == 0:
        messagebox.showerror("Error", "No valid images or points found in the selected folder.")
        return None
    if len(allPoints) != len(images):
        messagebox.showerror("Error", "Number of images and points files do not match.")
        return None

    eyecornerDst = [(int(0.3 * w), int(h / 3)), (int(0.7 * w), int(h / 3))]
    imagesNorm = []
    pointsNorm = []
    boundaryPts = np.array([
        (0, 0), (w // 2, 0), (w - 1, 0), (w - 1, h // 2),
        (w - 1, h - 1), (w // 2, h - 1), (0, h - 1), (0, h // 2)
    ])
    pointsAvg = np.zeros((len(allPoints[0]) + len(boundaryPts), 2), np.float32)
    numImages = len(images)

    for i in range(numImages):
        points1 = allPoints[i]
        eyecornerSrc = [points1[36], points1[45]]  # landmarks for eyes
        tform = similarityTransform(eyecornerSrc, eyecornerDst)
        img = cv2.warpAffine(images[i], tform, (w, h))
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (68, 2)))
        points = np.append(points, boundaryPts, axis=0)
        pointsAvg = pointsAvg + points / numImages
        pointsNorm.append(points)
        imagesNorm.append(img)

    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, pointsAvg)
    output = np.zeros((h, w, 3), np.float32())

    for i in range(numImages):
        img = np.zeros((h, w, 3), np.float32())
        for j in range(len(dt)):
            tin = []
            tout = []
            for k in range(3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)
                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)
                tin.append(pIn)
                tout.append(pOut)
            warpTriangle(imagesNorm[i], img, tin, tout)
        output = output + img

    output = output / numImages
    return output

def select_folder():
    # Use a temporary root for the dialog, then destroy it
    temp_root = tk.Tk()
    temp_root.withdraw()
    temp_root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images and Points", parent=temp_root, mustexist=True)
    temp_root.destroy()
    if folder_path:
        output = process_images(folder_path)
        if output is not None:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imshow('Morphed Image', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        messagebox.showinfo("No folder selected", "Please select a folder to proceed.")

def main():
    root = tk.Tk()
    root.title("Face Morphing Application")
    root.geometry("400x200")
    ttk.Label(root, text="Select a folder containing images and points:").pack(pady=20)
    ttk.Button(root, text="Browse", command=select_folder).pack(pady=10)
    root.mainloop()

if __name__ == '__main__':
    main()
