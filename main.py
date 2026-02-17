import tkinter as tk
from interface import AudioEditorApp # Importam interfata

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEditorApp(root)
    root.mainloop()