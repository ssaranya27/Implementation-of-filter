## EXPT 5:Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
</br>
Import cv2, numpy, and matplotlib.pyplot libraries.
</br> 

### Step2
</br>
Read the input image and convert it from BGR to RGB format.
</br> 

### Step3
</br>
Apply sharpening using a Laplacian kernel with cv2.filter2D() or using cv2.Laplacian().
</br> 

### Step4
</br>
Convert the result to an appropriate display format (np.uint8 if needed).
</br> 

### Step5
</br>
Display the original and sharpened images side by side using matplotlib.pyplot.imshow().
</br> 

## Program:
### Developed By   : SARANYA S.
### Register Number: 212223220101
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
# In[1]:Using Averaging Filter
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("flower.jpg")   # <-- Replace with your image name
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32) / 25
image3 = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()

```
ii) Using Weighted Averaging Filter
```Python
# In[2]:Using Weighted Averaging Filter
kernel1 = np.array([[1,2,1],
                    [2,4,2],
                    [1,2,1]], np.float32) / 16
image4 = cv2.filter2D(image2, -1, kernel1)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image4)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()


```
iii) Using Gaussian Filter
```Python
# In[3]:Using Gaussian Filter
gaussian_blur = cv2.GaussianBlur(image2, (5,5), 0)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

```
iv)Using Median Filter
```Python

# In[4]:Using Median Filter
median = cv2.medianBlur(image2, 5)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Filter Image")
plt.axis("off")
plt.show()

```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
# In[4]: Using Laplacian Kernel

import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("red.jpg")     # <-- replace with your image name
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Apply averaging filter first (for comparison)
kernel = np.ones((11,11), np.float32) / 121
image3 = cv2.filter2D(image2, -1, kernel)

# Laplacian sharpening kernel
kernel2 = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])

sharpened_img = cv2.filter2D(image2, -1, kernel2)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharpened_img)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()


```
ii) Using Laplacian Operator
```Python
# In[5]: Using Laplacian Operator

laplacian = cv2.Laplacian(image2, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))   # Convert to displayable form

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()

```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
</br>
<img width="900" height="721" alt="image" src="https://github.com/user-attachments/assets/f75c46dd-1a98-4581-9ec4-11b1aa282d77" />

</br>

ii)Using Weighted Averaging Filter
</br>
<img width="820" height="673" alt="image-1" src="https://github.com/user-attachments/assets/02fffce6-ea09-4f77-837b-ff60cc0345a5" />

</br>

iii)Using Gaussian Filter
</br>
<img width="821" height="682" alt="image-2" src="https://github.com/user-attachments/assets/19334ae1-6218-4710-b7be-dad787a41fd4" />
</br>
iv) Using Median Filter
</br>
<img width="792" height="687" alt="image-3" src="https://github.com/user-attachments/assets/8002495b-282c-493d-9571-4477aeab29ad" />
</br>

### 2. Sharpening Filters
</br>
i) Using Laplacian Kernal
</br>
<img width="828" height="667" alt="image-4" src="https://github.com/user-attachments/assets/465c06bc-032d-410b-a78d-00442de01ea7" />
</br>
ii) Using Laplacian Operator
</br>
<img width="807" height="690" alt="image-5" src="https://github.com/user-attachments/assets/e42e9354-e98b-4856-9b8f-8ab31d0abfd3" />
</br>

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
