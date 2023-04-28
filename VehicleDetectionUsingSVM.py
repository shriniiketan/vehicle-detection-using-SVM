
''' IMPORTING THE REQUIRED LIBRARIES '''

import numpy as np 
import cv2 
import matplotlib.image as mpimg
from skimage.feature import hog
import glob 

''' GETTING THE TRAINING DATASET '''

car = glob.glob('C:\SelfDrivingMaterials\Section6\data\car\**\*.png')
no_car = glob.glob('no car\**\*.png')


car_len = len(car) 
no_car_len = len(no_car)

print(car_len)
print(no_car_len)

''' HOG FEATURES FOR CAR IMAGES '''

car_hog_accum = []
blurr_kernel = np.ones((3,3)) *1/9

for i in car :
    image_color = mpimg.imread(i)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.filter2D(image_gray,-1,blurr_kernel)
    
    car_hog_feature, car_hog_img = hog(blurred_image,
                                       orientations = 11,
                                       pixels_per_cell =(16,16),
                                       cells_per_block=(2,2),
                                       transform_sqrt=False,
                                       visualize=True,
                                       feature_vector=True)
   
    car_hog_accum.append(car_hog_feature)
    

X_car = np.vstack(car_hog_accum).astype(np.float64)
Y_car = np.ones(len(X_car))

print(X_car.shape) 
print(Y_car.shape)


''' HOG FEATURE FOR NON CAR IMAGES '''


nocar_hog_accum = []
blurr_kernel = np.ones((3,3)) *1/9


for i in no_car :
    image_color = mpimg.imread(i)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.filter2D(image_gray,-1,blurr_kernel)
    
    nocar_hog_feature, nocar_hog_img = hog(blurred_image,
                                       orientations = 11,
                                       pixels_per_cell =(16,16),
                                       cells_per_block=(2,2),
                                       transform_sqrt=False,
                                       visualize=True,
                                       feature_vector=True)
   
    nocar_hog_accum.append(nocar_hog_feature)
    

X_nocar = np.vstack(nocar_hog_accum).astype(np.float64)
Y_nocar = np.zeros(len(X_nocar))

print(X_nocar.shape) 
print(Y_nocar.shape)

X = np.vstack((X_car,X_nocar))
Y = np.hstack((Y_car,Y_nocar))

print(X.shape)
print(Y.shape)


''' TRAINING THE SVM CLASSIFIER '''


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state = 101)
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report 

svc_model = LinearSVC()
svc_model.fit(X_train,Y_train)

Y_predict = svc_model.predict(X_test)
print(classification_report(Y_test, Y_predict))


''' OPTMISATION OF THE C AND GAMMA PARAMETERS FOR RBF KERNEl

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train,Y_train)


grid_predictions = grid.predict(X_test)
print(classification_report(Y_test,grid_predictions))

'''

test_image = mpimg.imread('C:\SelfDrivingMaterials\Section6\my_test_image_resized.jpg')
test_image = test_image.astype(np.float32)/255

pixels_in_cell = 16
HOG_orientations = 11
cells_in_block = 2
cells_in_step = 3

resizing_factor = 2
masked_region_shape = test_image.shape
L = masked_region_shape[1]/resizing_factor
W = masked_region_shape[0]/resizing_factor

masked_region_resized = cv2.resize(test_image, (np.int(L), np.int(W)))
masked_region_resized_R = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
print(masked_region_resized.shape)


print(masked_region_resized_R.shape)    
masked_region_hog_feature_all, hog_img = hog(masked_region_resized_R, 
                                             orientations = 11, 
                                             pixels_per_cell = (16, 16), 
                                             cells_per_block = (2, 2), 
                                             transform_sqrt = False,
                                             visualize = True, 
                                             feature_vector = False)
 



n_blocks_x = (masked_region_resized_R.shape[1] // pixels_in_cell)+1  
n_blocks_y = (masked_region_resized_R.shape[0] // pixels_in_cell)+1

#nfeat_per_block = orientations * cells_in_block **2 
blocks_in_window = (64 // pixels_in_cell)-1 
    
steps_x = (n_blocks_x - blocks_in_window) // cells_in_step
steps_y = (n_blocks_y - blocks_in_window) // cells_in_step

rectangles_found = []

for xb in range(steps_x):
    for yb in range(steps_y):
        y_position = yb*cells_in_step
        x_position = xb*cells_in_step
            
        hog_feat_sample = masked_region_hog_feature_all[y_position : y_position + blocks_in_window, x_position : x_position + blocks_in_window].ravel()
        x_left = x_position * pixels_in_cell
        y_top = y_position * pixels_in_cell
        print(hog_feat_sample.shape)  
        
        # predict using trained SVM
        test_prediction = svc_model.predict(hog_feat_sample.reshape(1,-1))
        # test_prediction = grid.predict(hog_feat_sample.reshape(1,-1))


    if test_prediction == 1: 
            rectangle_x_left = np.int(x_left * resizing_factor)
            rectangle_y_top = np.int(y_top * resizing_factor)
            window_dim = np.int(64 * resizing_factor)
            rectangles_found.append(((rectangle_x_left, rectangle_y_top),(rectangle_x_left + window_dim, rectangle_y_top + window_dim)))
                
Image_with_Rectangles_Drawn = np.copy(test_image)
    
for rectangle in rectangles_found:
    cv2.rectangle(Image_with_Rectangles_Drawn, rectangle[0], rectangle[1], (0, 255, 0), 20)



cv2.imwrite('OD1.jpg', Image_with_Rectangles_Drawn)














    
    







