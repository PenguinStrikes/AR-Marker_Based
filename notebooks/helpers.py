import cv2
import math
import numpy as np

def find_squares(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_b = cv2.GaussianBlur(img_g, (7, 7), 0)
    edges = cv2.Canny(img_b, 50, 150)

    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    for con in contours[1]:
        # This is the contour perimeter, True=closed shape
        perimeter = cv2.arcLength(con, True)
        # Create an approximatation of our contour
        approx = cv2.approxPolyDP(con, 0.1 * perimeter ,True)

        if len(approx) == 4:
            # Compute the bounding box of the approximate rectangle
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Calculate area and solidity
            area = cv2.contourArea(con)
            area_h = cv2.contourArea(cv2.convexHull(con))
            solidity = area / float(area_h)

            test_ar = aspect_ratio >= 0.7 and aspect_ratio <= 1.4
            test_dm = w > 50 and h > 50
            test_sl = solidity > 0.9

            if test_ar and test_dm and test_sl:
                img_box = cv2.drawContours(img, [approx], -1, (0,0,255), 5)
                                    
    return img

def marker_setup():
    # Load in our marker image
    img = cv2.imread('../data/img/test_image03.png',0)
    
    sift = cv2.xfeatures2d.SIFT_create() 
    kpm, desm = sift.detectAndCompute(img, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    # Match the marker descriptors with the scene descriptors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
            
    return img, sift, kpm, desm, flann

def find_markers(scene, img, sift, kpm, desm, flann, obj=None, render=False):
    # Create key points and descriptors
    kps, dess = sift.detectAndCompute(scene, None)
    
    if dess is not None:
        if len(dess) > 5:  
            # Create a matcher object
            
            matches = flann.knnMatch(desm, dess, k=2)

            # Sort the matches in order of distance
            good = []
            for m,n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
                    
            if len(good) > 20:
                src_pts = np.float32([kpm[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h,w = img.shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

                if (pts is not None) and (M is not None):
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    if render == True:
                        camera_parameters =  np.array([[800,0,320],[0,800,240],[0,0,1]])
                        projection = projection_matrix(camera_parameters, M)
                        scene = renders(scene, obj, projection, img, False)
                    else:
                        scene = cv2.polylines(scene, [np.int32(dst)], True, 255, 4, cv2.LINE_AA)
                        
    return scene


def renders(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h,w = model.shape
    
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (210,181,159))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    return img

def projection_matrix(camera_parameters, homography):
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)   
