from tkinter import *
import net
import torch
import torchvision
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def visualize(image):
    plt.imshow(image)
    plt.show()


def checkered(canvas, line_distance):
    # vertical lines at an interval of "line_distance" pixel
    for x in range(line_distance, canvas_width, line_distance):
        canvas.create_line(x, 0, x, canvas_height, fill="#476042")
    # horizontal lines at an interval of "line_distance" pixel
    for y in range(line_distance, canvas_height, line_distance):
        canvas.create_line(0, y, canvas_width, y, fill="#476042")


def draw_on_imagePIL(draw, shape, fillcolor, outlinecolor):
    draw.rectangle(shape, fill=fillcolor, outline=outlinecolor)


def paint(event):
    square_side = 30
    x1, y1 = round((event.x - square_side / 2) / square_side) * square_side, round(
        (event.y - square_side / 2) / square_side) * square_side
    x2, y2 = round((event.x + square_side / 2) / square_side) * square_side, round(
        (event.y + square_side / 2) / square_side) * square_side
    fillcolor, outlinecolor = 'black', 'black'
    w.create_rectangle(x1, y1, x2, y2, fill='black', outline="black")
    shape = [(x1, y1), (x2, y2)]
    draw_on_imagePIL(draw, shape, fillcolor, outlinecolor)
    #im = transform(image1.convert("L"))
    #im = transform(image1)[1]
    #tensor = im.unsqueeze(0)

    # visualize(transform(image1)[0])
    #prd = model(tensor)
    #print(prd.detach().numpy())
    #print(f" prediction = {prd.argmax().item()}")
    # draw.rectangle(shape, fill='black',outline='black')

def get_prediction_from_model():
    im = transform(image1.convert("L"))
    tensor = im.unsqueeze(0)
    softMax = torch.nn.Softmax()
    probs = (softMax(model(tensor)).detach().numpy())
    return probs



    # visualize(transform(image1)[0])
    #prd = model(tensor)
def clear():
    w.delete("all")
    checkered(w, square_side)
    image1.paste("white", box=(0, 0, width, height))


def visualize_image():
    visualize(image1.convert("L"))
    visualize(transform(image1)[0])
    visualize(transform(image1)[1])
    visualize(transform(image1)[2])

def plot_dist():
    x, y =[ i for i in range(10)], get_prediction_from_model()
    plt.bar(x, y)
    plt.xticks(x)
    ax = plt.gca()

    ax.set_ylim([0, 1])
    plt.show()


state = torch.load("MNIST0.984_net2")
model = net.Net2()
model.load_state_dict(state)
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28), antialias=True),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))])

master = Tk()
square_side = 30
image_sidex, image_sidey = 28, 28
canvas_width = image_sidex * square_side
canvas_height = image_sidey * square_side
#master.geometry(f"{canvas_width}x{canvas_height}")
w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
width = canvas_width
height = canvas_height
image1 = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image1)
w.pack()
checkered(w, square_side)
w.bind("<Button-1>", paint)
w.bind("<B1-Motion>", paint)
w.bind("<ButtonRelease-1>", paint)
master.title("Draw digits")
plot_display = Button(text="Get Distribution", command=plot_dist)
plot_display.pack()
#plot_display.grid(ipadx=10, ipady=10)
button_clear = Button(text="Clear All", command=clear)
button_clear.pack()
button_display = Button(text="Display", command=visualize_image)
#button_display.pack()
"""for c in range(3): master.columnconfigure(index=c, weight=1)
master.rowconfigure(index=0, weight=1)
plot_display.grid(row=0, column=0)
button_clear.grid(row=0, column=1)
button_display.grid(row=0, column=1)"""


mainloop()
