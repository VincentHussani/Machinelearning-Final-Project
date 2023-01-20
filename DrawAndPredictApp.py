from keras.api._v2.keras.models import load_model
from tkinter import *
from threading import *
import tkinter as tk
import win32gui
import glob
import os
import cv2
from PIL import ImageGrab, Image
import numpy as np
from time import *
import pickle
import sklearn
from sklearn import svm
from skimage.feature import hog
from tkinter import filedialog as fd
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Loads a starting model into the application, must be in the models folder
model = load_model("models/mod128w.h5", compile=False)


# Clears the folder where all images are saved
def clear_folder():
    files = glob.glob('processing/*')
    for f in files:
        os.remove(f)

# Initialized the application with some standard values


def get_handle():
    toplist = []
    windows_list = []
    canvas = 0

    def enum_win(hwnd, result):
        win_text = win32gui.GetWindowText(hwnd)
        windows_list.append((hwnd, win_text))
    win32gui.EnumWindows(enum_win, toplist)
    for (hwnd, win_text) in windows_list:
        if 'tk' == win_text:
            canvas = hwnd
    return canvas


class App(tk.Tk):
    def __init__(self):
        # Sets some initail values and saves some important information
        self.changed = False
        self.model = model
        self.gset = []
        self.history = []
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.update()

        bgcolor = "LavenderBlush2"
        self.configure(bg=bgcolor)

        # Creating the canvas and buttons
        self.canvas = tk.Canvas(
            self, width=1000, height=500, bg="white", cursor="cross")
        self.label = tk.Label(
            self, text="Draw!", bg=bgcolor, font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognise", bg="grey", command=self.get_canvas)
        self.button_clear = tk.Button(
            self, text="Clear", bg="grey", command=self.clear_all)
        self.button_clear_pics = tk.Button(
            self, text="Clear history", bg="grey", command=self.clear_history)
        self.button_open_file = tk.Button(
            self, text='Classify file', bg="grey", command=self.open_file)
        self.button_open_model = tk.Button(
            self, text='Choose model', bg="grey", command=self.open_model)
        # Grid structure
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.canvas.grid(row=0, column=0, padx=15, pady=15, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.button_open_file.grid(row=1, column=2, padx=4)
        self.button_open_model.grid(row=2, column=2, padx=4)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.button_clear_pics.grid(row=2, column=1, pady=1, padx=0.5)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    # This function does all the preprocessing on the images needed for the models to work
    def preprocessing_image(self):
        """function to preprocess the image to"""
        # The image to be preprocessed is loaded in
        image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('processing/startingimg.jpg', image)

        # Selects a threshhold and finds the contours in our image
        thresh_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(
            thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_imgs = []

        contours = contours[0] if len(contours) == 2 else contours[1]
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            # Crops the image according to the values found by the contours
            cropped_im = image[y:y+h, x:x+w]
            cv2.imwrite(
                f"processing/{len(processed_imgs)}croppedim.jpg", cropped_im)
            w_pad = 20  # Adds 20 pixels in the x axis
            h_pad = 20  # Adds 20 pixels in the y axis

            # For the models to work the picture needs to be sqaure here we find out how much we need to pad in what direction
            if h > w:
                w_pad += (h-w)//2
            elif w > h:
                h_pad += (w-h)//2

            # Here the padding is performed according to our previous calculations
            new_im = cv2.copyMakeBorder(
                cropped_im.copy(), h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=255)
            cv2.imwrite(
                f"processing/{len(processed_imgs)}paddedimg.jpg", new_im)

            # Image is resized to 28x28 and the colors are reversed
            new_im = cv2.resize(new_im, (28, 28))
            new_im = abs(255-new_im)

            cv2.imwrite(
                f"processing/{len(processed_imgs)}processedimg.jpg", new_im)

            if isinstance(self.model, sklearn.svm._classes.SVC) == 1:
                s = self.gset
                # Grabs a proccesed image and converts it to HOG features
                hogim = cv2.imread(
                    f"processing/{len(processed_imgs)}processedimg.jpg", cv2.IMREAD_GRAYSCALE)
                new_im = hog(hogim, orientations=s[0], pixels_per_cell=(s[1], s[1]),
                             cells_per_block=(s[2], s[2]), block_norm='L2', visualize=False)

            # Appens the final image to the list of processed images
            processed_imgs.append((new_im, x))
        # Sorts the images by position on the canvas
        processed_imgs = sorted(processed_imgs, key=lambda x: x[1])
        for i in range(len(processed_imgs)):
            processed_imgs[i] = processed_imgs[i][0]
        return processed_imgs

    def predict_digit(self, img):
        """function to predict the digit.
        Argument of function is PIL Image"""
        # A temporary image which is used to predict is created
        img.save('test.jpg')
        preprocessed_image = self.preprocessing_image()

        results = []
        # SVM and CNN require different functions to predict so here we check which on we have
        if isinstance(self.model, sklearn.svm._classes.SVC) == 1:
            for i in preprocessed_image:
                result = self.model.predict_proba([i])[0]
                indices = np.argpartition(result, -3)[-3:]
                results.append(list(zip(result[indices], indices)))
        else:
            for i in preprocessed_image:
                i = i.reshape(1, 28, 28, 1)
                result = self.model.predict([i], verbose=0)[0]
                indices = np.argpartition(result, -3)[-3:]
                results.append(list(zip(result[indices], indices)))

        os.remove('test.jpg')
        return results

    # Clears everything drawn on the canvas
    def clear_all(self):
        self.canvas.delete("all")

    # Lets the user select a model from a file on thier computer
    def open_model(self):
        # file type
        filetypes = (
            ('h5 files', '*.h5'),
            ('All files', '*.*')
        )

        # Get the path of the choosen file
        path = fd.askopenfilename(initialdir="/")
        # Tries to load the model with pickle
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            spot = path.find("SVM")
            svm_name = path[spot:]
            self.gset = list(map(int, re.findall(r'\d+', svm_name)))
        # If it fails to load Tensorflows load_model is used, this is an ugly solution but works.
        except:
            model = load_model(path, compile=False)
        # Saves what model we are currently using
        self.model = model

    # Lets the user select an image to predict from thier computer
    def open_file(self):
        # file type
        filetypes = (
            ('jpg files', '*.jpg'),
            ('All files', '*.*')
        )
        # Gets the path to the requested file
        f = fd.askopenfilename(initialdir="/", filetypes=filetypes)
        img = cv2.imread(f)
        img = Image.fromarray(img)
        self.classify_handwriting(img)

    # This function gets information the canvas proportion and location so it can be grabbed as an image.
    def get_canvas(self):
        clear_folder()
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        hwnd = get_handle()
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        x1, y1, x2, y2 = rect
        im = ImageGrab.grab((x1, y1, x2, y2))
        self.classify_handwriting(im)

    # Formats the results in a presentable manner and diplays it
    def classify_handwriting(self, im):
        results = self.predict_digit(im)
        COLUMNS = 4
        text = ""
        word = ""
        letter_num = 0

        # How many rows we need is calculated by the following formula then loops through everything for each row
        rows = len(results)//COLUMNS
        for row in range(rows+1):
            for letter in range(COLUMNS):
                if letter_num >= len(results):
                    break
                text += f"Letter {letter_num}:\t"
                letter_num += 1

            text += '\n'

            # Adds the most probable letter and it's % to be displayed
            for letter in range(COLUMNS*(row), COLUMNS*(row+1)):
                if letter >= len(results):
                    break
                sorted_result = sorted(
                    results[letter], key=lambda x: x[0], reverse=True)
                text += chr(sorted_result[0][1] + ord('A')) + \
                    ', ' + str(int(sorted_result[0][0]*100))+'%\t'
                word += chr(sorted_result[0][1] + ord('A'))

            text += '\n'
            # Adds the second most probable letter and it's % to be displayed
            for letter in range(COLUMNS*(row), COLUMNS*(row+1)):
                if letter >= len(results):
                    break
                sorted_result = sorted(
                    results[letter], key=lambda x: x[0], reverse=True)
                text += chr(sorted_result[1][1] + ord('A')) + \
                    ', ' + str(int(sorted_result[1][0]*100))+'%\t'

            text += '\n'
            # Adds the third most probable letter and it's % to be displayed
            for letter in range(COLUMNS*(row), COLUMNS*(row+1)):
                if letter >= len(results):
                    break
                sorted_result = sorted(
                    results[letter], key=lambda x: x[0], reverse=True)
                text += chr(sorted_result[2][1] + ord('A')) + \
                    ', ' + str(int(sorted_result[2][0]*100))+'%\t'
            text += '\n\n'

        # Displays some important information
        text += f"Word is: {word} "
        mdel = "CNN"
        if isinstance(self.model, sklearn.svm._classes.SVC) == 1:
            mdel = "SVM"
        self.history.append((mdel, word))
        text += "\n\n"
        text += "History"
        text += "\n"
        for i in reversed(self.history):
            text += f" {i[0]}: {i[1]}\n"

        # Sends the text to be displayed to the label
        self.label.configure(text=text, font=("Arial", 12))

    # Function for drawing
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y

        brush_size = 11

        self.canvas.create_oval(
            self.x-brush_size, self.y-brush_size, self.x + brush_size, self.y + brush_size, fill='black')
        self.changed = True

    # Clears saved prediction and refreshes the canvas
    def clear_history(self):
        self.history = []
        self.get_canvas()


# Creates the application
app = App()

# Runs the tkiner event loop
mainloop()
