import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Memuat gambar
image = cv2.imread('image.jpg')
# Mengonversi gambar dari BGR (default OpenCV) ke RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mengubah gambar menjadi array 2D dari piksel
pixels = image.reshape((-1, 3))
# Mengonversi tipe data ke float
pixels = np.float32(pixels)

# Menentukan jumlah kluster (k)
k = 3

# Menerapkan algoritma KMeans
kmeans = KMeans(n_clusters=k, max_iter=100, random_state=42)
kmeans.fit(pixels)

# Mendapatkan pusat kluster
centers = kmeans.cluster_centers_
# Mendapatkan label untuk setiap piksel
labels = kmeans.labels_

# Mengonversi pusat kluster ke uint8
centers = np.uint8(centers)
# Memetakan label ke pusat kluster yang sesuai
segmented_image = centers[labels.flatten()]

# Mengubah bentuk kembali ke bentuk gambar asli
segmented_image = segmented_image.reshape(image.shape)

# Memplot gambar asli dan gambar yang telah disegmentasi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Gambar Asli')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Gambar Tersegmentasi dengan k={}'.format(k))
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
