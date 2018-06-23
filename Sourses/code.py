import numpy as np
import cv2
def detectAndDescribe(image):
	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, features) = descriptor.detectAndCompute(image, None)
	kps = np.float32([kp.pt for kp in kps])
	return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	matches = []
	for m in rawMatches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))
		if len(matches) > 100:
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			return (matches, H, status)
	return None

def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
	temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
	list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

	[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

	translation_dist = [-x_min, -y_min]
	H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

	output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
	output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
	return output_img


image1 = cv2.imread("photo3.jpg",0)
image2 = cv2.imread("photo4.jpg",0)
(kps1, features1) = detectAndDescribe(image1)
(kps2, features2) = detectAndDescribe(image2)

ratio=0.8
reprojThresh=4.0
M21 = matchKeypoints(kps2, kps1, features2, features1, ratio, reprojThresh)

if M21 is None:
	print("M is None")
(matches, H, status) = M21

result = warpImages(image1, image2, H)
cv2.imshow("image 1",image1)
cv2.imshow("image 2",image2)
cv2.imshow("result",result)

cv2.waitKey(0)
cv2.destroyAllWindows()