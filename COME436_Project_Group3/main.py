import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn import svm
from skimage.feature import hog
from skimage import exposure


def prepare_data(image_paths, labels):
    features = []
    valid_labels = []
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Unable to load image at {img_path}")
            continue

        image = cv2.resize(image, (128, 128))
        feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(feature)
        valid_labels.append(labels[image_paths.index(img_path)])
    return np.array(features), np.array(valid_labels)



def train_model(features, labels):
    if len(features) == 0:
        raise ValueError("No valid images found for training. Please check your image paths.")
    classifier = svm.SVC(kernel='linear')
    classifier.fit(features, labels)
    return classifier



def classify_image(model, image):
    image = cv2.resize(image, (128, 128))
    feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    prediction = model.predict([feature])
    return prediction



class ClassificationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Electrical Component Classification")
        self.label = tk.Label(master, text="Select and Classify Images")
        self.label.pack()

        self.button = tk.Button(master, text="Upload Image", command=self.load_image)
        self.button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()


        image_paths = ['resistor.jpg', 'inductor.jpg', 'capacitor.jpg']
        labels = ['Resistor', 'Inductor', 'Capacitor']
        features, labels = prepare_data(image_paths, labels)
        try:
            self.model = train_model(features, labels)
        except ValueError as e:
            self.result_label.config(text=str(e))
            self.model = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path and self.model:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                prediction = classify_image(self.model, image)
                self.result_label.config(text=f"Classification Result: {prediction[0]}")
                cv2.imshow("Selected Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Error: Unable to load image at {file_path}")


root = tk.Tk()
app = ClassificationApp(root)
root.mainloop()
