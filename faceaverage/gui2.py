import os
import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

def readPoints(path):
    pointsArray = []
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".txt"):
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
        if filePath.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filePath))
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
    if point[0] < rect[0] or point[1] < rect[1] or point[0] > rect[2] or point[1] > rect[3]:
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri

def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p

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
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

def main():
    root = tk.Tk()
    root.title("Face Morphing - Average Face Generator")
    root.configure(bg="#282828")  # Midnight grey background

    # Style the buttons
    style = ttk.Style()

    # Configure style for TButton
    style.configure("TButton",
                    padding=10,
                    relief=tk.RAISED,  # Raised border
                    background="black",  # Black background
                    foreground="black",  # White text color
                    font=("Arial", 15),  # Bold font
                    width=20,  # Set button width
                    bordercolor="black",  # White border color
                    borderwidth=7,  # Border width
                    extra={"border-radius": 30})

    style.map("TButton",
              background=[("active", "#444444")],  # Darker background when active
              relief=[("active", tk.SUNKEN)])  # Sunken relief when active

    # Add a large heading text centered in x and y directions
    heading_label = tk.Label(root, text="Unleash your creativity with the power to create...", font=("Helvetica", 60, "bold"), fg="white", bg="#282828", wraplength=800)
    heading_label.pack(padx=10, pady=(230, 30), anchor="center")  # Padding for top and bottom, centered vertically

    # Create a frame to center align buttons vertically
    button_frame = tk.Frame(root, bg="#282828")
    button_frame.pack(expand=True, fill=tk.BOTH)  # Fill the entire window with the frame

    # Create the "Select Folder" and "Generate" buttons
    select_button = ttk.Button(button_frame, text="Select Folder", command=lambda: upload_and_process())
    select_button.pack(pady=20, anchor="center")

    generate_button = ttk.Button(button_frame, text="Generate", command=lambda: display_output())
    generate_button.pack(pady=10, anchor="center")


    def upload_and_process():
        global allPoints, images, pointsAvg, imagesNorm, pointsNorm, output
        path = filedialog.askdirectory(title='Select Folder Containing Images and Points')
        if not path:
            print("No folder selected. Exiting.")
            return
        w, h = 600, 600
        allPoints = readPoints(path)
        images = readImages(path)
        eyecornerDst = [(int(0.3 * w), int(h / 3)), (int(0.7 * w), int(h / 3))]
        imagesNorm, pointsNorm = [], []
        boundaryPts = np.array([(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)])
        pointsAvg = np.zeros((len(allPoints[0]) + len(boundaryPts), 2), np.float32)
        numImages = len(images)
        for i in range(numImages):
            points1 = allPoints[i]
            eyecornerSrc = [allPoints[i][36], allPoints[i][45]]
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
        dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))
        output = np.zeros((h, w, 3), np.float32)
        for i in range(len(imagesNorm)):
            img = imagesNorm[i]
            points = pointsNorm[i]
            for j in range(len(dt)):
                tin = []
                tout = []
                for k in range(3):
                    pIn = points[dt[j][k]]
                    pIn = constrainPoint(pIn, w, h)
                    pOut = pointsAvg[dt[j][k]]
                    pOut = constrainPoint(pOut, w, h)
                    tin.append(pIn)
                    tout.append(pOut)
                warpTriangle(img, output, tin, tout)
        # Save the output image
        cv2.imwrite("output.jpg", output * 255)

    def display_output():
        select_button.pack_forget()
        generate_button.pack_forget()
        heading_label.pack_forget()

        # Display the output image on the canvas
        output_img = cv2.imread("output.jpg")
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(output_img))

        # Create a label to display the image
        img_label = tk.Label(root, image=photo, bg="white")
        img_label.photo = photo  # Keep a reference to the image
        img_label.pack()

        # Center the image in the window
        def center_image(event=None):
            window_width = root.winfo_width()
            window_height = root.winfo_height()
            image_width = img_label.winfo_width()
            image_height = img_label.winfo_height()
            x_center = (window_width - image_width) // 2
            y_center = (window_height - image_height) // 2
            img_label.place(x=x_center, y=y_center)

        # Call the center_image function to initially center the image
        root.update_idletasks()  # Ensure all widgets are properly updated
        center_image()

        # Bind the <Configure> event to the center_image function
        root.bind('<Configure>', center_image)

    root.mainloop()

if __name__ == "__main__":
    main()
