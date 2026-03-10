import myclass
import myfile
import myplot
import myfilter
import myprocessing as myproc
import myextract
import cv2 as cv
import numpy as np

data_1d_original = None
data_1d_processed = None
data_2d_original = None
data_2d_processed = None
array_images_original = []
array_images_processed = []

def show_main_menu():
    print("Menu:")
    print("1. Read/Write Data")
    print("3. Apply filters")
    print("4. Extract features")
    print("5. Detect Defects")
    print("6. Pore Classification")
    while True:
        try:
            choice = int(input("Enter your selection: "))
            process_main_menu(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            show_main_menu()
            continue

def process_main_menu(choice):
    if choice == 1:
        show_sub_menu1();
    elif choice == 3:
        show_sub_menu3();
    elif choice == 4:
        show_sub_menu4();
    elif choice == 5:
        show_sub_menu5();
    elif choice == 6:
        show_sub_menu6();
    else:
        print("Invalid selection!")
        show_main_menu();

def show_sub_menu1():
    print("Read/Write Data:")
    print("1. Read and plot data from a csv file")
    print("2. Write the processed data to a csv file")
    print("3. Read and show an image")
    print("4. Save the processed image to an image file")
    print("5. Read all images in a folder")
    print("6. Save all processed images in a folder")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection: "))
            process_sub_menu1(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            continue

def process_sub_menu1(choice):
    global data_1d_original
    global data_1d_processed
    global data_2d_original
    global data_2d_processed
    global array_images_original
    global array_images_names

    if choice == 1:
        print("[Start] Read and plot data from a csv file")
        data_1d_original = myfile.read_csv_file()
        myplot.plot_1d_data(data_1d_original)
        print("[End] Read and plot data from a csv file")
        show_main_menu()
    elif choice == 2:
        print("[Start] Write data to a csv file")
        if data_1d_processed is None:
            print("[Error] The processed data buffer is blank.")
        else:
            myfile.save_csv_file(data_1d_processed)
            myplot.plot_1d_data(data_1d_processed)
        print("[End] Write data to a csv file")
        show_main_menu()
    elif choice == 3:
        print("[Start] Read and show image from a image file")
        data_2d_original = myfile.read_image_file()
        myplot.show_image(data_2d_original)
        print("[End] Read and show image from a image file")
        show_main_menu()
    elif choice == 4:
        print("[Start] Save the processed image to an image file")
        if data_2d_processed is None:
            print("[Error] The processed image buffer is None.")
        else:
            myfile.save_image_file(data_2d_processed)
            myplot.show_image(data_2d_processed)
        print("[End] Save the processed image to an image file")
        show_main_menu()
    elif choice == 5:
        print("[Start] Read all images in a folder")
        array_images_original, array_images_names = myfile.read_image_folder()
        print("[End] Read all images in a folder")
        show_main_menu()
    elif choice == 6:
        print("[Start] Save all processed images in a folder")
        if array_images_processed is None:
            print("[Error] The processed data buffer is blank.")
        else:
            myfile.save_images_to_folder(array_images_processed, array_images_names)
        print("[End] Save all processed images in a folder")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu1()

def show_sub_menu3():
    print("Apply filters:")
    print("1. Apply Gaussian filter on 1D data")
    print("2. Apply Gaussian filter on an image")
    print("3. Apply Gaussian filter on all loaded images")
    print("4. Apply Sobel filter on an image")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection: "))
            process_sub_menu3(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            continue

def process_sub_menu3(choice):
    global data_1d_original
    global data_1d_processed
    global data_2d_original
    global data_2d_processed
    global array_images_original
    global array_images_processed

    if choice == 1:
        print("[Start] Apply Gaussian filter on 1D data")
        data_1d_processed = myfilter.gaussian_filter(data_1d_original, 1.5)
        myplot.plot_two_1d_data(data_1d_original, data_1d_processed)
        print("[End] Apply Gaussian filter on 1D data")
        show_main_menu()
    elif choice == 2:
        print("[Start] Apply Gaussian filter on image")
        data_2d_processed = cv.GaussianBlur(data_2d_original, (5,5), cv.BORDER_DEFAULT)
        myplot.show_two_images(data_2d_original, data_2d_processed)
        print("[End] Apply Gaussian filter on image")
        show_main_menu()
    elif choice == 3:
        print("[Start] Apply Gaussian filter all loaded images")
        array_images_processed = []
        for entry in array_images_original:
            array_images_processed.append(cv.GaussianBlur(entry, (5, 5), cv.BORDER_DEFAULT))
        print("[End] Apply Gaussian filter all loaded images")
        show_main_menu()
    elif choice == 4:
        print("[Start] Apply Sobel filter on image")
        data_2d_processed = cv.Sobel(data_2d_original, cv.CV_64F, 1, 0, ksize=5)
        myplot.show_two_images(data_2d_original, data_2d_processed)
        print("[End] Apply Sobel filter on image")
        show_main_menu()
    elif choice == 5:
        print("[Start] Apply Sobel filter all loaded images")
        array_images_processed = []
        for entry in array_images_original:
            array_images_processed.append(cv.Sobel(entry, cv.CV_64F, 1, 0, ksize=5))
        print("[End] Apply Sobel filter all loaded images")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu1()

def show_sub_menu4():
    print("Extract features:")
    print("1. Min Intensity")
    print("2. Max Intensity")
    print("3. Mean Intensity")
    print("4. Average Intensity")
    print("5. Standard Deviation")
    print("6. Spectral Entropy")
    print("7. Extract all Features")
    print("8. Extract all Features")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection: "))
            process_sub_menu4(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            continue

def process_sub_menu4(choice):
    global min_intensity
    global max_intensity
    global mean_intensity
    global average_intensity
    global standard_deviation
    global spectral_entropy


    if choice == 1:
        print("[Start] Extract Min Intensity")
        min_intensity = myextract.calculate_min_intensity(data_2d_original)
        print("Min Intensity:", min_intensity)
        print("[End] Extract Min Intensity")
        show_main_menu()
    elif choice == 2:
        print("[Start] Extract Max Intensity")
        max_intensity = myextract.calculate_max_intensity(data_2d_original)
        print("Max Intensity:", max_intensity)
        print("[End] Extract Max Intensity")
        show_main_menu()
    elif choice == 3:
        print("[Start] Extract Mean Intensity")
        mean_intensity = myextract.calculate_mean_intensity(data_2d_original)
        print("Mean Intensity:", mean_intensity)
        print("[End] Extract Mean Intensity")
        show_main_menu()
    elif choice == 4:
        print("[Start] Extract Average Intensity")
        average_intensity = myextract.calculate_average_intensity(data_2d_original)
        print("Average Intensity:", average_intensity)
        print("[End] Extract Average Intensity")
        show_main_menu()
    elif choice == 5:
        print("[Start] Extract Standard Deviation")
        standard_deviation = myextract.calculate_standard_deviation(data_2d_original)
        print("Standard Deviation:", standard_deviation)
        print("[End] Extract Standard Deviation")
        show_main_menu()
    elif choice == 6:
        print("[Start] Extract Spectral Entropy")
        spectral_entropy = myextract.calculate_spectral_entropy(data_2d_original)
        print("Spectral Entropy:", spectral_entropy)
        print("[End] Extract Spectral Entropy")
        show_main_menu()
    elif choice == 7:
        min_intensity, max_intensity, mean_intensity, average_intensity, standard_deviation, spectral_entropy = myextract.extract_features(data_2d_original)
        print("Min Intensity:", min_intensity)
        print("Max Intensity:", max_intensity)
        print("Mean Intensity:", mean_intensity)
        print("Average Intensity:", average_intensity)
        print("Standard Deviation:", standard_deviation)
        print("Spectral Entropy:", spectral_entropy)
        print("[End] Extract all Features")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu1()

def show_sub_menu5():
    print("Detect Defects:")
    print("1. LCD defect detection using pattern comparison")
    print("2. LCD defect detection using FFT")
    print("3. Pore detection using thresholding")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection: "))
            process_sub_menu5(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            continue

def process_sub_menu5(choice):
    global data_1d_original
    global data_1d_processed
    global data_2d_original
    global data_2d_processed
    global array_images_original
    global array_images_processed

    if choice == 1:
        print("[Start] LCD defect detection using pattern comparison")
        if data_2d_original is None:
            print("[Error] The image buffer is None.")
        else:
            pattern_period = myproc.find_pattern_period(data_2d_original)
            data_2d_processed = myproc.eliminate_patterns(data_2d_original, pattern_period)
        print("[End] LCD defect detection using pattern comparison")
        show_main_menu()
    elif choice == 2:
        print("[Start] LCD defect detection using FFT")
        data_2d_processed = myproc.detect_defects_fft(data_2d_original)
        print("[End] LCD defect detection using FFT")
        show_main_menu()
    elif choice == 3:
        print("[Start] Pore detection using thresholding")
        # Step 2: Pore Detection (Segmentation)
        data_2d_processed = myproc.segment_pores(data_2d_original)
        # Step 3: Feature Extraction
        features = myproc.extract_features(data_2d_processed)
        # Prepare labeled data (features and labels) - assuming you have extracted features and labeled pores as 1, non-pores as 0
        labels = np.array([])  # Placeholder, replace with your actual labels
        # Step 4: Supervised Machine Learning
        model, x_test, y_test = myproc.train_model(features, labels)
        # Evaluate the model's performance
        accuracy = myproc.evaluate_model(model, x_test, y_test)
        print(f"Accuracy: {accuracy}")
        print("[End] Pore detection using thresholding")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu1()

def show_sub_menu6():
    print("Pore Classification:")
    print("1. Apply Vector Machine method")
    print("2. Apply Random Forest method")
    print("3. Apply Decision Tree method")
    print("4. Apply CNN-based method")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection: "))
            process_sub_menu6(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            continue

def process_sub_menu6(choice):

    if choice == 1:
        print("[Start] Apply Vector Machine method")
        myclass.class_vector()
        print("[End] Apply Vector Machine method")
        show_main_menu()
    elif choice == 2:
        print("[Start] Apply Random Forest method")
        myclass.class_random()
        print("[End] Apply Random Forest method")
        show_main_menu()
    elif choice == 3:
        print("[Start] Apply Decision Tree method")
        myclass.class_decision()
        print("[End] Apply Decision Tree method")
        show_main_menu()
    elif choice == 4:
        print("[Start] Apply CNN-based method")
        myclass.class_cnn()
        #myclass.class_cnn2()
        print("[End] Apply CNN-based method")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu1()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    show_main_menu()
