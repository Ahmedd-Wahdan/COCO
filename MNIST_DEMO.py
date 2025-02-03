import tkinter as tk
import numpy as np
from core.utils import load_model


model = load_model()

class MNISTGUI:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")
        
       
        self.canvas_size = 560  # 28x28 grid with 20x20 pixels per cell
        self.cell_size = self.canvas_size // 28
        self.brush_size = 1  # More precise drawing
        
        
        self.data = np.zeros((28, 28), dtype=np.float32)
        
       
        self.create_canvas()
        self.create_probability_bars()
        self.create_clear_button()
        
        
        self.last_x = None
        self.last_y = None
        
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)
    
    def create_canvas(self):
        self.canvas = tk.Canvas(self.master, width=self.canvas_size, 
                               height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
       
        self.rects = []
        for i in range(28):
            row = []
            for j in range(28):
                x0 = i * self.cell_size
                y0 = j * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, 
                                                   fill='white', outline='lightgray')
                row.append(rect)
            self.rects.append(row)
    
    def create_probability_bars(self):
        self.prob_frame = tk.Frame(self.master)
        self.prob_frame.grid(row=0, column=1, padx=10, pady=10)
        
        self.labels = []
        for i in range(10):
            frame = tk.Frame(self.prob_frame, height=30, width=200, 
                            highlightbackground='black', highlightthickness=1)
            frame.pack(pady=2, fill=tk.X)
            
            label = tk.Label(frame, text=f"{i}: 0.00%", 
                            font=('Arial', 12), anchor='w')
            label.pack(fill=tk.X)
            self.labels.append(label)
    
    def create_clear_button(self):
        self.clear_btn = tk.Button(self.master, text='Clear', 
                                  command=self.clear_canvas)
        self.clear_btn.grid(row=1, column=0, columnspan=2, pady=10)
    
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
        for dx in range(2):  # 2 columns (current and next)
            for dy in range(2):  # 2 rows (current and next)
                if 0 <= x+dx < 28 and 0 <= y+dy < 28:  # Check bounds
                    self.canvas.itemconfig(self.rects[x+dx][y+dy], fill='black')
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
        input_data = self.data.reshape(1,1, 28, 28)
        predictions = model.predict(input_data)[0]
        
        # Update probability labels
        for i, prob in enumerate(predictions):
            self.labels[i].config(text=f"{i}: {prob*100:.2f}%")
    
    def clear_canvas(self):
        # Reset data array and canvas
        self.data.fill(0)
        for row in self.rects:
            for rect in row:
                self.canvas.itemconfig(rect, fill='white')
        for label in self.labels:
            label.config(text=f"{label['text'][0]}: 0.00%")
        self.reset_last()
    
    def reset_last(self, event=None):
        self.last_x = None
        self.last_y = None

if __name__ == "__main__":
    root = tk.Tk()
    gui = MNISTGUI(root)
    root.mainloop()