import tkinter as tk
import numpy as np
from core.utils import load_model

model = load_model('mnist_coco.pkl')  

# Define a set of colors for each digit's prediction bar
DIGIT_COLORS = {
    0: "#e57373",  # red
    1: "#f06292",  # pink
    2: "#ba68c8",  # purple
    3: "#9575cd",  # deep purple
    4: "#7986cb",  # indigo
    5: "#64b5f6",  # blue
    6: "#4fc3f7",  # light blue
    7: "#4dd0e1",  # cyan
    8: "#4db6ac",  # teal
    9: "#81c784"   # green
}

class MNISTGUI:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")
        
        self.canvas_size = 560  # 28x28 grid with 20x20 pixels per cell
        self.cell_size = self.canvas_size // 28
        
        self.data = np.zeros((28, 28), dtype=np.float32)
        
        self.create_canvas()
        self.create_probability_canvas()  # Modified appearance
        self.create_clear_button()
        
        self.last_x = None
        self.last_y = None
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)
    
    def create_canvas(self):
        # Use a light gray background and a sunken border for a modern look
        self.canvas = tk.Canvas(
            self.master, width=self.canvas_size, 
            height=self.canvas_size, bg='#f0f0f0', bd=3, relief='sunken'
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        self.rects = []
        for i in range(28):
            row_rects = []
            for j in range(28):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                # Use a soft outline color for the grid
                rect = self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill='#ffffff', outline='#d3d3d3'
                )
                row_rects.append(rect)
            self.rects.append(row_rects)
    
    def create_probability_canvas(self):
        """
        Creates smaller prediction boxes arranged horizontally.
        Each box has the digit displayed above it with distinct colors.
        """
        self.prob_frame = tk.Frame(self.master, bg='#ffffff')
        self.prob_frame.grid(row=1, column=0, padx=10, pady=10)
        
        # Define bar dimensions (smaller than before)
        self.bar_width = 60
        self.bar_height = 20
        
        # Dictionary to store each digit's (canvas, fill rectangle id)
        self.prob_items = {}
        for digit in range(10):
            # Create a sub-frame for each digit's display and bar.
            sub_frame = tk.Frame(self.prob_frame, bg='#ffffff')
            sub_frame.pack(side=tk.LEFT, padx=5)
            
            # Label for the digit (placed above the probability box)
            lbl = tk.Label(sub_frame, text=str(digit), font=('Arial', 12, 'bold'),
                           fg=DIGIT_COLORS[digit], bg='#ffffff')
            lbl.pack(side=tk.TOP)
            
            # Canvas for the probability bar with a visible border.
            bar_canvas = tk.Canvas(
                sub_frame, width=self.bar_width, height=self.bar_height, 
                bg='#e0f7fa', bd=1, relief='solid'
            )
            bar_canvas.pack(side=tk.TOP, pady=3)
            
            # Create the fill rectangle (initially with 0 width) using the digit's color.
            fill_rect = bar_canvas.create_rectangle(
                0, 0, 0, self.bar_height, fill=DIGIT_COLORS[digit], width=0
            )
            
            # Save the canvas and fill rectangle ID for updating later.
            self.prob_items[digit] = (bar_canvas, fill_rect)
    
    def create_clear_button(self):
        self.clear_btn = tk.Button(
            self.master, text='Clear', command=self.clear_canvas,
            font=('Arial', 12, 'bold'), bg="#ff8a65", fg="white", bd=2, relief='raised'
        )
        self.clear_btn.grid(row=2, column=0, columnspan=1, pady=10)
    
    def paint(self, event):
        # Convert mouse coordinates to grid coordinates
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        
        # Paint cells between previous and current position for smooth drawing
        if self.last_x is not None and self.last_y is not None:
            for cell in self.get_line_cells(self.last_x, self.last_y, x, y):
                self.paint_cell(*cell)
        
        self.paint_cell(x, y)
        self.last_x = x
        self.last_y = y
        self.predict()
    
    def paint_cell(self, x, y):
        # Paint a 2x2 square with (x, y) as the top-left corner
        for dx in range(2):
            for dy in range(2):
                if 0 <= x+dx < 28 and 0 <= y+dy < 28:
                    # Use a deep charcoal color for drawn pixels
                    self.canvas.itemconfig(self.rects[y+dy][x+dx], fill='#333333')
                    self.data[y+dy][x+dx] = 1.0
    
    def get_line_cells(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells
    
    def predict(self, event=None):
        # Reshape data to (1, 1, 28, 28)
        input_data = self.data.reshape(1, 1, 28, 28)
        predictions = model.predict(input_data)[0]
        
        # Update each probability box based on its prediction.
        for digit, prob in enumerate(predictions):
            bar_canvas, fill_rect = self.prob_items[digit]
            fill_width = self.bar_width * prob
            # Update the fill rectangle's width to reflect the predicted probability.
            bar_canvas.coords(fill_rect, 0, 0, fill_width, self.bar_height)
    
    def clear_canvas(self):
        # Reset data array and canvas
        self.data.fill(0)
        for row in self.rects:
            for rect in row:
                self.canvas.itemconfig(rect, fill='#ffffff')
        for digit in self.prob_items:
            bar_canvas, fill_rect = self.prob_items[digit]
            # Reset the fill rectangle to 0 width.
            bar_canvas.coords(fill_rect, 0, 0, 0, self.bar_height)
        self.reset_last()
    
    def reset_last(self, event=None):
        self.last_x = None
        self.last_y = None

if __name__ == "__main__":
    root = tk.Tk()
    gui = MNISTGUI(root)
    root.mainloop()
