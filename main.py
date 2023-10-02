import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from classifier import ResumeClassifier

classifier = ResumeClassifier()

def open_file():
    file_path = filedialog.askopenfilename()
    with open(file_path, 'r', encoding='latin-1') as file:
        resume_text = file.read()

    prediction = classifier.predict(resume_text)
    result_label.config(text=f"Result: {prediction}")

root = tk.Tk()
root.title("Resume Screening System")

# Load and display the background image
bg_image = Image.open('data/sample.jpg')  # Replace 'background_image.jpg' with your image file
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

open_button = tk.Button(root, text="Upload Resume", command=open_file, font=("Arial", 16, "bold"))
open_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

result_label = tk.Label(root, text="Result: ", font=("Arial", 18, "bold"))
result_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Load the trained classifier
classifier.train('data/UpdatedResumeDataSet.csv')

root.mainloop()
