import numpy as np
from tkinter import *
from PIL import Image, ImageEnhance
import numpy.linalg
import imageio.v2 as imageio
import time
import matplotlib.pyplot as plt
import cv2
import random
from scipy import signal

# Function to perform Singular Value Decomposition (SVD) and measure execution time
def DSV(B, k):
    start_time = time.time()  # Start time for measuring execution time
    U, Sigma, V = np.linalg.svd(B.copy())  # Perform SVD
    print("* Execution time in seconds is", (time.time() - start_time), "*")  # Print execution time
    return U, Sigma, V  # Return decomposed matrices

# Function to perform SVD and plot the singular values and their cumulative sum
def DSV2(B, k):
    U, Sigma, V = np.linalg.svd(B.copy())  # Perform SVD
    plt.semilogy(Sigma)  # Plot singular values on a semilogarithmic scale
    plt.show()
    plt.plot(np.cumsum(Sigma) / np.sum(Sigma))  # Plot the cumulative sum of singular values
    plt.show()
    return U, Sigma, V  # Return decomposed matrices

# Function to generate graphics for each color channel using SVD
def grafice(image, k):
    r = image[:, :, 0]  # Extract red channel
    g = image[:, :, 1]  # Extract green channel
    b = image[:, :, 2]  # Extract blue channel

    # Apply SVD and plot for each color channel
    ur, sr, vr = DSV2(r, k)
    ug, sg, vg = DSV2(g, k)
    ub, sb, vb = DSV2(b, k)

# Function to display the original image
def show_images(image_name):
    plt.title("Original_image.jpg")
    plt.imshow(image_name)  # Display image
    plt.axis('on')  # Show axis
    plt.show()

# Function to compress an image using SVD
def compress_image(image, k):
    original_bytes = image.nbytes  # Calculate original image size in bytes
    print("Space (in bytes) to store this image is =", original_bytes)
    image = image / 255  # Normalize image
    linie, coloana, _ = image.shape  # Get image dimensions

    # Split the image into 3 2D matrices for each color channel
    Red = image[:, :, 0]  # Red channel
    Green = image[:, :, 1]  # Green channel
    Blue = image[:, :, 2]  # Blue channel

    # Perform SVD on each color channel
    U_Red, Sigma_Red, V_Red = DSV(Red, k)
    U_Green, Sigma_Green, V_Green = DSV(Green, k)
    U_Blue, Sigma_Blue, V_Blue = DSV(Blue, k)

    # Calculate total size of all matrices in bytes
    bytes_matrices = sum([matrix.nbytes for matrix in [U_Red, Sigma_Red, V_Red, U_Green, Sigma_Green,
                                                       V_Green, U_Blue, Sigma_Blue, V_Blue]])
    print("The stored matrices have a total size (in bytes) of", bytes_matrices)

    # Truncate matrices to use only the first k components
    U_red_k = U_Red[:, 0:k]
    V_red_k = V_Red[0:k, :]
    Sigma_Red_k = Sigma_Red[0:k]

    U_green_k = U_Green[:, 0:k]
    V_green_k = V_Green[0:k, :]
    Sigma_Green_k = Sigma_Green[0:k]

    U_blue_k = U_Blue[:, 0:k]
    V_blue_k = V_Blue[0:k, :]
    Sigma_Blue_k = Sigma_Blue[0:k]

    # Calculate total size of compressed matrices
    compressedBytes = sum([matrix.nbytes for matrix in [U_red_k, Sigma_Red_k, V_red_k, U_green_k,
                                                        Sigma_Green_k, V_green_k, U_blue_k, Sigma_Blue_k, V_blue_k]])
    print("The compressed matrices we want to store have a size (in bytes) of", compressedBytes)

    # Calculate compression ratio
    rata = 100 * compressedBytes / original_bytes
    print("The compression rate is", rata, "% of the original")

    # Reconstruct approximate images for each color channel
    image_red_aproximation = np.matrix(U_Red[:, :k]) * np.diag(Sigma_Red[:k]) * np.matrix(V_Red[:k, :])
    image_green_aproximation = np.matrix(U_Green[:, :k]) * np.diag(Sigma_Green[:k]) * np.matrix(V_Green[:k, :])
    image_blue_aproximation = np.matrix(U_Blue[:, :k]) * np.diag(Sigma_Blue[:k]) * np.matrix(V_Blue[:k, :])

    # Combine color channels to form compressed image
    compressed_image = np.zeros((linie, coloana, 3))
    compressed_image[:, :, 0] = image_red_aproximation
    compressed_image[:, :, 1] = image_green_aproximation
    compressed_image[:, :, 2] = image_blue_aproximation

    np.clip(compressed_image, 0, 255, out=compressed_image)  # Ensure pixel values are in the correct range

    plt.title("Compressed Image")
    plt.imshow(compressed_image)  # Display compressed image
    plt.axis('off')  # Hide axis
    return compressed_image

# Main menu function to select image operations
def menu():
    path = input("Please select an image= ")
    image = imageio.imread(path)
    print("**Welcome to Compress Images**")
    print()

    # Options for various operations
    choice = input("""
                      1: Show the Image
                      2: Red color image
                      3: Green color image
                      4: Blue color image 
                      5: Compress the RGB image
                      6: Show Graphics
                      7: Compress the image with 4 different values
                      8: Adjust Image Brightness
                      9: Add and remove noise from Image
                      10: Exit

                   Please enter your choice: """)

    # Conditional statements to perform selected operation
    if choice == "1":
        image = image / 255  # Normalize image for display
        rand, coloana, _ = image.shape
        print("Pixels ", rand, "X", coloana)
        show_images(image)
    elif choice == "2":
        # Display red color channel of the image
        path = input("Please select an image= ")
        image = Image.open(path)
        red_band = image.getdata(band=0)
        img_mat = np.array(list(red_band), float)
        img_mat.shape = (image.size[1], image.size[0])
        img_mat = np.matrix(img_mat)
        plt.title("Red Channel Image")
        plt.imshow(img_mat)
        plt.axis('on')
        plt.show()
    elif choice == "3":
        # Display green color channel of the image
        path = input("Please select an image= ")
        image = Image.open(path)
        green_band = image.getdata(band=1)
        img_mat = np.array(list(green_band), float)
        img_mat.shape = (image.size[1], image.size[0])
        img_mat = np.matrix(img_mat)
        plt.title("Green Channel Image")
        plt.imshow(img_mat)
        plt.axis('on')
        plt.show()
    elif choice == "4":
        # Display blue color channel of the image
        path = input("Please select an image= ")
        image = Image.open(path)
        blue_band = image.getdata(band=2)
        img_mat = np.array(list(blue_band), float)
        img_mat.shape = (image.size[1], image.size[0])
        img_mat = np.matrix(img_mat)
        plt.title("Blue Channel Image")
        plt.imshow(img_mat)
        plt.axis('on')
        plt.show()
    elif choice == "5":
        # Compress the RGB image using SVD
        rand, coloana, _ = image.shape
        print("Pixels ", rand, "X", coloana)
        k = int(input("Enter the value of k= "))
        compressed_image = compress_image(image, k)
        plt.title("Compressed Image")
        plt.imshow(compressed_image)
        plt.axis('on')
        plt.show()
    elif choice == "6":
        # Show graphics for SVD analysis of the image
        k = int(input("Enter the value of k= "))
        grafice(image, k)
    elif choice == "7":
        # Compress the image with 4 different k values for comparison
        image1 = compress_image(image, 5)
        image2 = compress_image(image, 20)
        image3 = compress_image(image, 50)
        image4 = compress_image(image, 100)
        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        axs[0, 0].imshow(image1)
        axs[0, 0].set_title('Compressed Image: k=5', size=10)
        axs[0, 1].imshow(image2)
        axs[0, 1].set_title('Compressed Image: k=20', size=10)
        axs[1, 0].imshow(image3)
        axs[1, 0].set_title('Compressed Image: k=50', size=10)
        axs[1, 1].imshow(image4)
        axs[1, 1].set_title('Compressed Image: k=100', size=10)
        plt.tight_layout()
        plt.savefig('reconstructed_images_using_different_values.jpg', dpi=150)
        plt.show()
    elif choice == "8":
        # Adjust image brightness
        path = input("Please select an image= ")
        im = Image.open(path)
        enhancer = ImageEnhance.Brightness(im)
        factor = 1  # Original brightness
        im_output = enhancer.enhance(factor)
        im_output.save('original-image.png')
        plt.imshow(im_output)
        plt.show()
        factor = 0.3  # Darkened image
        im_output = enhancer.enhance(factor)
        im_output.save('darkened-image.png')
        plt.imshow(im_output)
        plt.show()
        factor = 2.8  # Brightened image
        im_output = enhancer.enhance(factor)
        plt.imshow(im_output)
        plt.show()
    elif choice == "9":
        # Add and remove noise from the image
        path = input("Please select an image= ")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        noisy = np.zeros(img.shape, np.uint8)

        p = 0.2  # Probability of noise

        # Traverse through the image pixels and add noise
        for i in range(img.shape[0]):  # Rows
            for j in range(img.shape[1]):  # Columns
                r = random.random()
                if r < p / 2:
                    noisy[i][j] = [0, 0, 0]  # Black noise
                elif r < p:
                    noisy[i][j] = [255, 255, 255]  # White noise
                else:
                    noisy[i][j] = img[i][j]  # Original image pixel

        # Apply median blur to denoise the image
        denoised = cv2.medianBlur(noisy, 5)

        output = [img, noisy, denoised]

        titles = ['Original', 'Noisy', 'Denoised']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(output[i])
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()
    elif choice == "10":
        # Exit the program
        exit()
    else:
        # Invalid choice, prompt the menu again
        print("Invalid choice, please try again.")
        menu()

# Main function to run the menu
def main():
    menu()

main()
