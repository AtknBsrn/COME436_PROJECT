import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

"""
Class labels for more understandable terms
"""
class_labels = {
    'Resistor': 'Resistor',
    'Inductor': 'Inductor',
    'Capacitor': 'Capacitor'
}

def load_data_from_folders(base_path):
    """
    Load image paths and labels from specified folders.
    """
    image_paths = []
    labels = []
    for label in ['Capacitor', 'Inductor', 'Resistor']:
        folder_path = os.path.join(base_path, label)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder_path, filename))
                labels.append([label])
    return image_paths, labels

def augment_image(image):
    """
    Augment the image by flipping and rotating it.
    """
    augmented_images = [image, cv2.flip(image, 1)]
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        augmented_images.append(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])))
    return augmented_images

def prepare_data(image_paths, labels):
    """
    Prepare data by loading images, applying augmentation, and extracting HOG features.
    """
    features = []
    valid_labels = []
    for img_path, img_labels in zip(image_paths, labels):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Unable to load image at {img_path}")
            continue

        image = cv2.resize(image, (128, 128))
        augmented_images = augment_image(image)
        for aug_image in augmented_images:
            feature = hog(aug_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            features.append(feature)
            valid_labels.append(img_labels[0])
    return np.array(features), np.array(valid_labels)

def train_model(features, labels):
    """
    Train the SVM model using the provided features and labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train.ravel())
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return classifier

def classify_image(model, image):
    """
    Classify a single image using the trained model.
    """
    image = cv2.resize(image, (128, 128))
    feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    prediction = model.predict([feature])
    return class_labels.get(prediction[0], prediction[0])

class ClassificationApp:
    def __init__(self, master):
        """
        Initialize the GUI application, train the model, and set up UI elements.
        """
        self.master = master
        self.master.title("Electrical Component Classification")
        self.master.geometry("1000x600")
        self.label = tk.Label(master, text="Select and Classify Images")
        self.label.pack()

        self.button = tk.Button(master, text="Upload Image", command=self.load_image)
        self.button.pack()

        self.exit_button = tk.Button(master, text="Exit", command=self.master.quit, bg="red", fg="white")
        self.exit_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.gray_canvas = tk.Canvas(master, width=420, height=460, bg='white')
        self.gray_canvas.pack(side=tk.LEFT, padx=10)
        self.gray_canvas.create_rectangle(10, 10, 410, 410, outline="black", width=2)
        self.gray_image_label = tk.Label(self.gray_canvas)
        self.gray_image_label.place(x=20, y=20)
        self.gray_canvas.create_text(210, 440, text="Image Gray Scale", font=("Arial", 12))

        self.color_canvas = tk.Canvas(master, width=420, height=460, bg='white')
        self.color_canvas.pack(side=tk.RIGHT, padx=10)
        self.color_canvas.create_rectangle(10, 10, 410, 410, outline="black", width=2)
        self.color_image_label = tk.Label(self.color_canvas)
        self.color_image_label.place(x=20, y=20)
        self.color_canvas.create_text(210, 440, text="Original Image", font=("Arial", 12))

        # Load and prepare the data
        base_path = '.'
        image_paths, labels = load_data_from_folders(base_path)
        features, labels = prepare_data(image_paths, labels)
        try:
            self.model = train_model(features, labels)
        except ValueError as e:
            self.result_label.config(text=str(e))
            self.model = None

    def load_image(self):
        """
        Load an image from file, classify it, and display the result.
        """
        file_path = filedialog.askopenfilename()
        if file_path and self.model:
            gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            color_image = cv2.imread(file_path)
            if gray_image is not None and color_image is not None:
                prediction = classify_image(self.model, gray_image)
                self.result_label.config(text=f"Classification Result: {prediction}")

                resized_gray_image = cv2.resize(gray_image, (400, 400))
                resized_gray_image = Image.fromarray(resized_gray_image)
                resized_gray_image = ImageTk.PhotoImage(resized_gray_image)
                self.gray_canvas.create_image(210, 210, image=resized_gray_image)
                self.gray_image_label.image = resized_gray_image

                resized_color_image = cv2.resize(color_image, (400, 400))
                resized_color_image = Image.fromarray(cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2RGB))
                resized_color_image = ImageTk.PhotoImage(resized_color_image)
                self.color_canvas.create_image(210, 210, image=resized_color_image)
                self.color_image_label.image = resized_color_image
            else:
                print(f"Error: Unable to load image at {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassificationApp(root)
    root.mainloop()
