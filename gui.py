import tkinter as tk, PIL, cv2, tensorflow as tf, keras.models, numpy as np
from PIL import Image, ImageDraw
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.datasets import mnist

classes=[0,1,2,3,4,5,6,7,8,9]

#   GUI components for drawing
def start():
    root = tk.Tk()
    root.title('Painting Canvas')

    #   paint method to draw a circle at each point travelled
    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill='black')
        draw.line([x1,y1,x2,y2], fill='black', width=40)

    image1 = PIL.Image.new("RGB", (400, 400), (255,255,255))
    draw =ImageDraw.Draw(image1)

    #*  Canvas components and event handling
    canvas = tk.Canvas(root, width=400, height=400)
    
    canvas.bind('<B1-Motion>', paint)

    #*  Text component to dynamically change the label component
    text = tk.StringVar()
    text.set('Click and drag to draw a letter')
    label = tk.Label(root, textvariable=text)

    #*  Button GUI components
    predict_button = tk.Button(root, text='Predict', command=lambda: predict(image1, text), padx=5, highlightbackground='blue')
    clear_button = tk.Button(root, text='Clear', command=lambda: clear(canvas, draw, text), padx=5, highlightbackground='blue',)
    
    label.pack(side=tk.TOP)
    canvas.pack(expand=True, fill=tk.BOTH)
    predict_button.pack(side=tk.LEFT, padx=10, pady=10)
    clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

    root.mainloop()

#   Command for the clear button
#   canvas  -Canvas gui component for painting
#   draw    -The image component for predicting
#   text    -Text component to change the label gui component
def clear(canvas, draw, text):
    canvas.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    text.set('')

#   Command for the predict button
#   image   -Raw image component to save the png
#   text    -Text component to change the label gui component
def predict(image, text):
    fileName = 'image.png'
    image.save(fileName)

    img = cv2.imread('image.png', 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img,(28,28))
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img = img/255.0

    pred = model.predict(img)
    print('\nargmax', np.argmax(pred[0]), '\n', pred[0][np.argmax(pred[0])])
    text.set('Prediction: ' + str(np.argmax(pred[0])) + '\tConfidence: ' + str( round( (pred[0][np.argmax(pred[0])] * 100), 2)) + '%')

#   Creates, compiles, trains, and saves the model file
def model():
    img_rows, img_cols = 28, 28
    num_classes = 10
    batch_size = 128
    epochs = 10

    #*  Load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    #*  Normalize RGB values
    x_train = x_train/255
    x_test = x_test/255

    #*  Create binary class matrix
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    #*  Add the model layers
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    #!  Compile and train the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #!  Save the model as a file
    model.save('digit_model.h5')

if __name__ == "__main__":
    model = tf.keras.models.load_model('digit_model.h5')
    start()