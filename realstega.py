#importing the required packages
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image
import cv2
import stego
import time
from sklearn.externals import joblib
from sklearn.externals.joblib import  load
import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
#function for brightness calculation of image
def calculate_brightness(file):
    image=Image.open(file)
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    brightcal = 1 if brightness == 255 else brightness / scale
    print("brightness calculation = "+str(brightcal))
    return brightcal
#function for contrast calculation of image
def contrast(file):
    img=cv2.imread(file)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = img_grey.std()
    print("contrast value of image = "+str(contrast))
    return contrast
#function for pixel calculation  of image
def get_num_pixels(file):
    width, height = Image.open(file).size
    numpix = width * height
    print("number of pixels = "+str(numpix))
    return numpix
#function for finding number of bits in text
def no_of_bits(textval):
    textget = str(textval.get("1.0", END))
    textbit = (len(textget)-1) * 8
    print("text obtained: "+textget, end='')
    print("number of bits = "+str(textbit))
    return textbit
#function for error handling for undefined variables
def error_handle(func):
    try:
        func
    except NameError:
        print("well, it WASN'T defined after all!")
    else:
        print("sure, it was defined.")
#gui window for hide operation
def hide_window():
    top = Toplevel()
    top.focus_force()
    top.title("Hide")
    width = 400
    height = 400
    top.resizable(False, False)
    top.iconbitmap(r"icon.ico")
    top.configure(bg="#102529")
    posRight = int(top.winfo_screenwidth() / 2 - width / 2)
    posDown = int(top.winfo_screenheight() / 2 - height / 2)
    top.geometry("{}x{}+{}+{}".format(width, height, posRight, posDown))
    label1 = Label(top, text="Enter Text You Want To Embed", fg="#ff0048",bg="#102529", font=("arial", 12, "bold"))
    label1.place(x=35, y=10)
    text=Text(top,height=5,width=40,wrap=WORD,fg="#91ff00",font=("arial", 11, "bold"), bg="black")
    text.config(state="normal")
    text.place(x=35,y=30)

    label2 = Label(top, text="Choose Your Cover Image ", fg="#ff0048", bg="#102529", font=("arial", 12, "bold"))
    label2.place(x=35, y=130)
    btnFind=Button(top,text="Select An Image",width=45,command=getOpenPath, fg="white", bg="#18635e")
    btnFind.place(x=35,y=155)

    label3 = Label(top, text="Save Your Destination Image", fg="#ff0048", bg="#102529", font=("arial", 12, "bold"))
    label3.place(x=35, y=200)
    btnFind = Button(top, text="Save Image As", width=45, command=getSavePath, fg="white", bg="#18635e")
    btnFind.place(x=35, y=225)
    '''ent=Entry(top,width=36,font=("arial", 12, "bold"))
    ent.config(state="disabled")
    ent.place(x=35,y=240)'''
    b1 = Button(top, text="Begin Hide", command= lambda:[encode_message(openpath, text, savepath)], width=10, fg="white", bg="#6c9400", relief="solid",font=("arial", 15, "bold"))
    b1.place(x=35, y=300)
    b2 = Button(top, text="Go Back", width=10, command=lambda:[top.destroy(),start_window()], fg="white", bg="#6c9400", relief="solid",font=("arial", 15, "bold"))
    b2.place(x=220, y=300)
    top.protocol('WM_DELETE_WINDOW', exit_window)
#function for starting encoding process
def encode_message(file, message, savefile):
    bits = no_of_bits(message)
    if len(file)>=1 and bits>=1 and len(savefile)>=1:
       print("IMAGE FILE IS LOADED")
       bright=calculate_brightness(file)
       con=contrast(file)
       pix=get_num_pixels(file)
       param=[[con, pix]]
       classifier = joblib.load('filename.pkl')
       #print(classifier)
       sc = load('std_scaler.bin')
       p=classifier.predict(sc.transform(param))
       l=stego.edge(file)
       sum=len(l)
       if p == 0 and sum<bits:
          print("IMAGE FILE NOT ACCEPTABLE")
          messagebox.showinfo("ERROR", "Image File not Acceptable.")
       else:
          print("IMAGE FILE ACCEPTABLE")
          # note time before encode
          t1 = time.time()
          stego.encode(file,str(message.get("1.0", END)),savefile)
          # note time after encoding
          t2 = time.time()
          # calculate time difference
          dt1 = t2 - t1
          # print the time difference
          print("total encoding time = " + str(round(dt1, 3)) + " secs")
          messagebox.showinfo("SUCCESS", "Encoding Completed Successfully in " + str(round(dt1, 3)) + " secs.")
    else:
        if bits<1:
            messagebox.showinfo("ERROR", "No Input Text was given for Encoding.")
        if len(file)<1:
            messagebox.showinfo("ERROR", "No Image File was Selected for Opening.")
        if len(savefile)<1:
            messagebox.showinfo("ERROR", "Save Location for the Image File was not Chosen.")
#gui window for unhide operation
def unhide_window():
    global disp
    top=Toplevel()
    top.focus_force()
    top.title("Unhide")
    width = 400
    height = 400
    top.resizable(False, False)
    top.iconbitmap(r"icon.ico")
    top.configure(bg="#102529")
    posRight = int(top.winfo_screenwidth() / 2 - width / 2)
    posDown = int(top.winfo_screenheight() / 2 - height / 2)
    top.geometry("{}x{}+{}+{}".format(width, height, posRight, posDown))
    label1 = Label(top, text="Choose Image You Want To Decode", fg="#ff0048", bg="#102529", font=("arial", 12, "bold"))
    label1.place(x=35, y=10)
    btnFind = Button(top, text="Select An Image", width=45, command=getOpenPath_PNG_BMP, fg="white", bg="#18635e")
    btnFind.place(x=35, y=35)
    b1 = Button(top, text="Begin Unhide", command = lambda:[decode_message(openpath_png)], width=11, fg="white", bg="#6c9400", relief="solid", font=("arial", 15, "bold"))
    b1.place(x=35, y=70)
    b2 = Button(top, text="Go Back", width=11, command=lambda:[top.destroy(),start_window()], fg="white", bg="#6c9400", relief="solid",font=("arial", 15, "bold"))
    b2.place(x=220, y=70)

    label1 = Label(top, text="The Embedded Text Is", fg="#ff0048", bg="#102529", font=("arial", 12, "bold"))
    label1.place(x=35, y=130)
    disp = Text(top, height=10, width=40, wrap=WORD, fg="#91ff00", font=("arial", 11, "bold"), bg="black")
    disp.config(state="normal")
    disp.place(x=35, y=150)
    top.protocol('WM_DELETE_WINDOW', exit_window)
#function for starting decoding process
def decode_message(file):
    if len(file)>=1:
       # note time before decode
       t1 = time.time()
       print("IMAGE FILE IS LOADED")
       msg=stego.decode(file)
       # note time after decoding
       t2 = time.time()
       # calculate time difference
       dt2 = t2 - t1
       print("decoded data: "+msg, end='')
       disp.delete(1.0, END)
       disp.insert("1.0", msg)
       # print the time difference
       print("total decoding time = " + str(round(dt2, 3)) + " secs")
       messagebox.showinfo("SUCCESS", "Decoding Completed Successfully in "+str(round(dt2, 3))+" secs.")
    else:
        messagebox.showinfo("ERROR", "No Image File was Selected for Opening.")
#gui function for opening file explorer for only png and bmp image types
def getOpenPath_PNG_BMP():
    '''global folderPath
    folder_selected=filedialog.askdirectory()
    folderPath.set(folder_selected)'''
    global openpath_png
    openpath_png=filedialog.askopenfilename(title="Open An Image", filetypes=(("PNG Images", "*.png"),
                                                                              ("BMP Images", "*.bmp"),
                                                                          ("All Files", "*.*")))
    if len(openpath_png)>=1:
        messagebox.showinfo("File Selected", "Selected Image File is at: "+openpath_png)
#gui function for opening file explorer for all images
def getOpenPath():
    '''global folderPath
    folder_selected=filedialog.askdirectory()
    folderPath.set(folder_selected)'''
    global openpath
    openpath=filedialog.askopenfilename(title="Open An Image", filetypes=(("JPG Images", "*.jpg"),
                                                 ("JPEG Images", "*.jpeg"),
                                                 ("PNG Images", "*.png"),
                                                 ("BMP Images", "*.bmp"),
                                                 ("All Files", "*.*")))
    if len(openpath)>=1:
        messagebox.showinfo("File Selected", "Selected Image File is at: "+openpath)
#gui function for saving file explorer
def getSavePath():
    '''global SavePath
    folder_selected=filedialog.askdirectory()
    SavePath.set(folder_selected)'''
    global savepath
    savepath=filedialog.asksaveasfilename(title="Save Image As", defaultextension='.png',
                                          filetypes=(("PNG Images", "*.png"),
                                                     ("BMP Images", "*.bmp"),
                                                    ("All Files", "*.*")))
    if len(savepath)>=1:
        messagebox.showinfo("Save Location Chosen", "Image File will be Saved in: "+savepath)
#gui window for choosing options
def start_window():
    top=Toplevel()
    top.focus_force()
    top.title("Welcome")
    width=400
    height=400
    top.resizable(False,False)
    top.iconbitmap(r"icon.ico")
    top.configure(bg="#102529")
    posRight = int(top.winfo_screenwidth() / 2 - width / 2)
    posDown = int(top.winfo_screenheight() / 2 - height / 2)
    top.geometry("{}x{}+{}+{}".format(width, height, posRight, posDown))
    label1=Label(top,text="Choose Your Option",fg="#ff0048",bg="#102529",font=("arial",16,"bold"))
    label1.place(x=100,y=50)
    b1=Button(top,text="Hide",width=10,command=lambda:[top.destroy(),hide_window()],fg="white",bg="#6c9400",relief="solid",font=("arial",16,"bold"))
    b1.place(x=130,y=100)

    b2 = Button(top, text="Unhide",width=10,command=lambda:[top.destroy(),unhide_window()], fg="white", bg="#6c9400", relief="solid", font=("arial", 16, "bold"))
    b2.place(x=130,y=150)

    b3 = Button(top, text="Return", width=10, command=lambda:[top.destroy(),window.deiconify()], fg="white", bg="#6c9400", relief="solid",font=("arial", 16, "bold"))
    b3.place(x=130, y=200)
    top.protocol('WM_DELETE_WINDOW', exit_window)
#gui function for about application
def about():
    messagebox.showinfo("About RealStega", "Development Team: Abhisek, Abhinaba, Abhishek, Anshuman\nAbhiMan Softworks © 2020 All rights reserved.")
#program console events display
print("----PROGRAM CONSOLE----")
#Creating the main tkinter gui window
window=Tk()
window.focus_force()
window.configure(bg="#102529")
image=Image.open(r"startup.png")
photo_image=ImageTk.PhotoImage(image)
label2=Label(window,image=photo_image)
label2.place(x=60,y=50)
window.resizable(False,False)
windowWidth = 400
windowHeight = 400
positionRight = int(window.winfo_screenwidth() / 2- windowWidth / 2)
positionDown = int(window.winfo_screenheight() / 2 - windowHeight / 2)
window.geometry("{}x{}+{}+{}".format(windowWidth,windowHeight,positionRight,positionDown))
window.iconbitmap(r"icon.ico")
window.title("RealStega 1.0")
label=Label(window,text="Image Steganography Tool",fg="#ff0048",bg="#102529",font=("arial",16,"bold"))
label.place(x=65,y=10)
button1=Button(window,text="Get Started",width=10,command=lambda:[window.withdraw(),start_window()],fg="white",bg="#6c9400",relief=RIDGE,font=("arial",16,"bold"))
button1.place(x=130,y=250)
button2=Button(window,text="Exit",width=10,command=window.destroy,fg="white",bg="#6c9400",relief=RIDGE,font=("arial",16,"bold"))
button2.place(x=130,y=300)
about = Button(window,text="About", width=5, fg="white", bg="#18635e", command = about, font=("arial",10,"bold"))
about.pack(side="bottom", anchor=E, )
#gu function for exit button title
def exit_window():
        window.destroy()
window.protocol('WM_DELETE_WINDOW', exit_window)
#kicking off the gui
window.mainloop()



