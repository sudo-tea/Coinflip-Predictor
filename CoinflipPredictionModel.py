import numpy as np
import tkinter as tk
from collections import deque
from tkinter import font

class NeuralNetwork:
    def __init__(self, input_size=3, hidden_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2  # Heads (0) or Tails (1)
        
        # Initialize weights with float32 precision
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1
        self.b1 = np.zeros((1, hidden_size), dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, self.output_size).astype(np.float32) * 0.1
        self.b2 = np.zeros((1, self.output_size), dtype=np.float32)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        # Ensure input is float32
        X = X.astype(np.float32)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, lr=0.1):
        m = X.shape[0]
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def train(self, X, y, epochs=1000, lr=0.1):
        # Convert to float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, lr)

class CoinFlipPredictor:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.model = NeuralNetwork(input_size=window_size)
        self.history = deque(maxlen=50)
    
    def create_training_data(self):
        X, y = [], []
        history_list = list(self.history)
        for i in range(len(history_list) - self.window_size):
            # Convert H/T to 0/1 for input features
            window = [0 if f == 'H' else 1 for f in history_list[i:i+self.window_size]]
            X.append(window)
            # Convert H/T to 0/1 for labels
            next_val = 0 if history_list[i+self.window_size] == 'H' else 1
            y.append(next_val)
        return np.array(X), np.eye(2, dtype=np.float32)[np.array(y)]
    
    def train_model(self):
        if len(self.history) > self.window_size:
            X, y = self.create_training_data()
            self.model.train(X, y, epochs=1000, lr=0.1)
    
    def predict_next(self):
        if len(self.history) >= self.window_size:
            # Convert last window to 0/1
            input_seq = [0 if f == 'H' else 1 for f in list(self.history)[-self.window_size:]]
            X = np.array([input_seq], dtype=np.float32)
            probs = self.model.forward(X)[0]
            return ('H', probs[0]) if np.argmax(probs) == 0 else ('T', probs[1])
        return ('N/A', 0)

class CoinFlipGUI:
    def __init__(self, master):
        self.master = master
        master.title("Coin Flip Predictor")
        self.predictor = CoinFlipPredictor()
        
        self.bold_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.large_font = font.Font(family="Helvetica", size=24, weight="bold")
        
        # History display
        self.history_label = tk.Label(master, text="Flip History:", font=self.bold_font)
        self.history_label.pack(pady=5)
        self.history_text = tk.Text(master, height=4, width=30, state=tk.DISABLED)
        self.history_text.pack()
        
        # Input buttons
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(pady=10)
        self.btn_heads = tk.Button(
            self.btn_frame, 
            text="Add Heads (H)", 
            command=lambda: [self.add_heads(), self.train_model()], 
            bg="lightgreen"
        )
        self.btn_heads.pack(side=tk.LEFT, padx=5)
        self.btn_tails = tk.Button(
            self.btn_frame, 
            text="Add Tails (T)", 
            command=lambda: [self.add_tails(), self.train_model()], 
            bg="salmon"
        )
        self.btn_tails.pack(side=tk.LEFT, padx=5)
        
        # Prediction display
        self.pred_frame = tk.Frame(master, relief=tk.RIDGE, borderwidth=2)
        self.pred_frame.pack(pady=15, padx=10, fill=tk.X)
        self.pred_header = tk.Label(self.pred_frame, text="Next Flip Prediction:", font=self.bold_font)
        self.pred_header.pack(side=tk.LEFT, padx=10)
        self.pred_display = tk.Label(self.pred_frame, text="N/A", font=self.large_font, width=6)
        self.pred_display.pack(side=tk.LEFT, padx=10)
        self.conf_display = tk.Label(self.pred_frame, text="Confidence: 0%", font=self.bold_font)
        self.conf_display.pack(side=tk.RIGHT, padx=10)
        
        # Control buttons
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=10)
        self.clear_btn = tk.Button(self.control_frame, text="Clear History", command=self.clear_history, bg="lightgrey")
        self.clear_btn.pack(side=tk.LEFT, padx=5)

    def update_display(self):
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, ' '.join(self.predictor.history))
        self.history_text.config(state=tk.DISABLED)
        prediction, confidence = self.predictor.predict_next()
        self.update_prediction_display(prediction, confidence)
    
    def update_prediction_display(self, prediction, confidence):
        if prediction == 'H':
            self.pred_display.config(text="HEADS", fg="darkgreen", bg="palegreen")
        elif prediction == 'T':
            self.pred_display.config(text="TAILS", fg="darkred", bg="mistyrose")
        else:
            self.pred_display.config(text="N/A", fg="black", bg="white")
        self.conf_display.config(text=f"Confidence: {confidence*100:.1f}%")
    
    def add_heads(self):
        self.predictor.history.append('H')
        self.update_display()
    
    def add_tails(self):
        self.predictor.history.append('T')
        self.update_display()
    
    def train_model(self):
        self.predictor.train_model()
        self.update_display()
    
    def clear_history(self):
        self.predictor.history.clear()
        self.update_display()
        self.pred_display.config(text="N/A", bg="white")
        self.conf_display.config(text="Confidence: 0%")

if __name__ == "__main__":
    root = tk.Tk()
    app = CoinFlipGUI(root)
    root.mainloop()
