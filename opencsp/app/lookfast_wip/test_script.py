import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk


class RectangleCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectangle Crop Selector")

        # Initialize variables
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.frame_image = None
        self.corners = []  # List to store the 4 corners
        self.rect_id = None  # ID for the rectangle drawn on the canvas
        self.corner_text_ids = []  # IDs for corner text annotations

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Load video button
        self.load_button = tk.Button(self.root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        # Canvas for displaying the video frame
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)  # Bind left mouse click
        self.canvas.bind("<Button-3>", self.reset_selection)  # Bind right mouse click
        self.canvas.bind("<Motion>", self.on_mouse_move)  # Bind mouse motion

        # Mouse position readout
        self.mouse_position_label = tk.Label(self.root, text="Mouse Position: (X: 0, Y: 0)")
        self.mouse_position_label.pack()

        # Confirm button
        self.confirm_button = tk.Button(self.root, text="Confirm Rectangle", command=self.confirm_rectangle)
        self.confirm_button.pack()

    def load_video(self):
        # Open file dialog to select video
        self.video_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if not self.video_path:
            return

        # Open video using OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        # Read the first frame
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read the first frame.")
            return

        self.current_frame = frame
        self.display_frame(frame)

    def display_frame(self, frame):
        # Convert the frame to a format suitable for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.frame_image = ImageTk.PhotoImage(image)

        # Display the frame on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame_image)

    def on_canvas_click(self, event):
        # Record the clicked position
        if len(self.corners) < 4:
            self.corners.append((event.x, event.y))
            self.display_corner(event.x, event.y)
            self.draw_rectangle()
        else:
            messagebox.showinfo("Info", "You have already selected 4 corners.")

    def reset_selection(self, event):
        # Reset the selection of corners and clear the canvas annotations
        self.corners = []
        self.corner_text_ids = []
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = None

        # Clear corner text annotations
        for text_id in self.corner_text_ids:
            self.canvas.delete(text_id)

        messagebox.showinfo("Info", "Selection reset. You can start over.")

    def on_mouse_move(self, event):
        # Update the mouse position readout
        self.mouse_position_label.config(text=f"Mouse Position: (X: {event.x}, Y: {event.y})")

    def display_corner(self, x, y):
        # Display the corner coordinates on the canvas
        corner_text = f"({x}, {y})"
        text_id = self.canvas.create_text(x + 10, y - 10, text=corner_text, fill="yellow", font=("Arial", 10))
        self.corner_text_ids.append(text_id)

    def draw_rectangle(self):
        # Draw the rectangle based on the selected corners
        if len(self.corners) == 2:
            x1, y1 = self.corners[0]
            x2, y2 = self.corners[1]
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
        elif len(self.corners) == 4:
            x1, y1 = self.corners[0]
            x2, y2 = self.corners[2]
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def confirm_rectangle(self):
        # Ensure 4 corners are selected
        if len(self.corners) != 4:
            messagebox.showerror("Error", "Please select 4 corners of the rectangle.")
            return

        # Calculate the bounding box
        x_coords = [corner[0] for corner in self.corners]
        y_coords = [corner[1] for corner in self.corners]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Display the bounding box coordinates
        messagebox.showinfo("Rectangle Coordinates", f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max})")

        # Save the coordinates for cropping
        self.crop_coordinates = (x_min, y_min, x_max, y_max)
        print(f"Crop coordinates saved: {self.crop_coordinates}")

        # Close the application
        self.root.destroy()

    def crop_video_frames(self):
        # Crop all frames of the video using the saved coordinates
        if not hasattr(self, "crop_coordinates"):
            messagebox.showerror("Error", "No crop coordinates saved.")
            return

        x_min, y_min, x_max, y_max = self.crop_coordinates

        # Create output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        # Open video again
        self.cap = cv2.VideoCapture(self.video_path)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Crop the frame
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Save the cropped frame
            output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            cv2.imwrite(output_path, cropped_frame)

        messagebox.showinfo("Success", f"All frames cropped and saved to {output_dir}")


if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = RectangleCropApp(root)
    root.mainloop()
