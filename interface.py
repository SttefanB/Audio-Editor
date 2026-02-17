import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import sounddevice as sd
from audio_processor import AudioProcessor
import sys

class AudioEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor Audio - V11 (STABIL)")
        self.root.geometry("1150x900")
        
        # --- FIX: Initializam variabilele grafice AICI (la inceput), nu la final ---
        self.right_panel = None
        self.analysis_fig = None    
        self.analysis_canvas = None 
        
        self.setup_styles()
        self.root.configure(bg="#2b2b2b")
        
        self.proc = AudioProcessor()
        self.current_file_path = None
        
        # --- AUDIO VARS ---
        self.play_pos = 0
        self.stream = None
        self.is_playing = False
        self.chunk_size = 1024
        self.spk_win = 2048
        self.spectrum_update_interval = 30
        self.needs_stop = False
        self.playback_buffer = None
        self.original_data_copy = None
        self._gui_after_id = None

        # --- TABS ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.tab_editor = tk.Frame(self.notebook, bg="#2b2b2b")
        self.notebook.add(self.tab_editor, text=" 🎛️ Editor Audio ")
        self.setup_editor_ui()

        self.tab_analysis = tk.Frame(self.notebook, bg="#dddddd")
        self.notebook.add(self.tab_analysis, text=" 📈 Analiză Parametrii ")
        
        # Aceasta functie va crea self.right_panel
        self.setup_analysis_ui() 

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # --- NOTA: Am sters liniile de aici care setau self.right_panel = None ---

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#2b2b2b", borderwidth=0)
        style.configure("TNotebook.Tab", background="#444", foreground="white", padding=[20, 10], font=("Arial", 11, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#2196F3")], foreground=[("selected", "white")])

    # --- EDITOR UI ---
    def setup_editor_ui(self):
        top_frame = tk.Frame(self.tab_editor, bg="#2b2b2b", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(top_frame, text="📂 Încarcă Fișier", command=self.load_file, 
                  bg="#444", fg="white", font=("Arial", 10, "bold"), padx=10).pack(side=tk.LEFT, padx=10)
        ttk.Separator(top_frame, orient="vertical").pack(side=tk.LEFT, fill="y", padx=10)
        tk.Button(top_frame, text="▶ Play", command=self.play_audio, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="⏸ Pauză", command=self.stop_audio, bg="#f44336", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="⏮ Reset", command=self.reset_cursor_to_start, bg="#FF9800", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        filter_frame = tk.LabelFrame(self.tab_editor, text="Filtre & Efecte", bg="#dddddd", fg="black", font=("Arial", 10, "bold"), pady=5)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        lp_frame = tk.Frame(filter_frame, bg="#dddddd")
        lp_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(lp_frame, text="Low Pass (Hz):", bg="#dddddd").pack(side=tk.LEFT)
        self.scale_lp = tk.Scale(lp_frame, from_=50, to=5000, orient=tk.HORIZONTAL, length=150, bg="#dddddd"); self.scale_lp.set(1000)
        self.scale_lp.pack(side=tk.LEFT, padx=5)
        tk.Button(lp_frame, text="Aplică LP", command=self.apply_low_pass_from_slider, bg="#673AB7", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT)

        ttk.Separator(filter_frame, orient="vertical").pack(side=tk.LEFT, fill="y", padx=10)
        
        hp_frame = tk.Frame(filter_frame, bg="#dddddd")
        hp_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(hp_frame, text="High Pass (Hz):", bg="#dddddd").pack(side=tk.LEFT)
        self.scale_hp = tk.Scale(hp_frame, from_=50, to=5000, orient=tk.HORIZONTAL, length=150, bg="#dddddd"); self.scale_hp.set(500)
        self.scale_hp.pack(side=tk.LEFT, padx=5)
        tk.Button(hp_frame, text="Aplică HP", command=self.apply_high_pass_from_slider, bg="#673AB7", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        
        ttk.Separator(filter_frame, orient="vertical").pack(side=tk.LEFT, fill="y", padx=10)
        tk.Button(filter_frame, text="📞 Telefon", command=self.apply_telephone, bg="#555", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(filter_frame, text="↩ Resetează la Original", command=self.restore_clean_original, bg="#607D8B", fg="white", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=10)

        self.fig_main = Figure(figsize=(9,6), dpi=100)
        self.ax_wave = self.fig_main.add_subplot(211); self.ax_wave.set_title("Forma de undă"); self.ax_wave.grid(True, alpha=0.5)
        self.ax_spec = self.fig_main.add_subplot(212); self.ax_spec.set_title("Spectru"); self.ax_spec.set_xlabel("Hz"); self.fig_main.tight_layout(pad=3.0)
        
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.tab_editor)
        self.canvas_main.draw()
        self.canvas_main.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.line_wave = None; self.cursor_line = None; self.spec_line = None; self.static_spec_line = None
        self.bg_wave = None; self.bg_spec = None

    # --- ANALIZA UI ---
    def setup_analysis_ui(self):
        left_panel = tk.Frame(self.tab_analysis, bg="#cfcfcf", width=220, padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)
        
        tk.Label(left_panel, text="Alege Parametru:", bg="#cfcfcf", font=("Arial", 12, "bold")).pack(pady=20)
        
        tk.Button(left_panel, text="1. Energie Scurtă", command=lambda: self.show_analysis_graph("energie"),
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold"), pady=10, width=20).pack(pady=5)
        
        tk.Button(left_panel, text="2. ZCR (Zero Cross)", command=lambda: self.show_analysis_graph("zcr"),
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), pady=10, width=20).pack(pady=5)
        
        tk.Button(left_panel, text="3. Autocorelație", command=lambda: self.show_analysis_graph("auto"),
                  bg="#9C27B0", fg="white", font=("Arial", 10, "bold"), pady=10, width=20).pack(pady=5)
        
        tk.Label(left_panel, text="-----------------", bg="#cfcfcf").pack(pady=10)
        tk.Button(left_panel, text="⬅ Înapoi la Editor", command=lambda: self.notebook.select(0),
                  bg="#666", fg="white", font=("Arial", 10), pady=5).pack(side=tk.BOTTOM, pady=20)

        # Containerul din dreapta - AICI se creeaza self.right_panel
        self.right_panel = tk.Frame(self.tab_analysis, bg="white")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.lbl_analiza_info = tk.Label(self.right_panel, text="Selectează un parametru din stânga.", 
                                         font=("Arial", 14), bg="white", fg="#555")
        self.lbl_analiza_info.place(relx=0.5, rely=0.5, anchor="center")

    def show_analysis_graph(self, param_type):
        # 1. Verificare date
        if self.proc.data is None:
            messagebox.showwarning("Atenție", "Nu ai încărcat niciun fișier în Editor!")
            return

        # 2. Curatare panou dreapta (DISTRUGEM VECHIUL CANVAS)
        # ACUM self.right_panel NU mai este None, deci nu va mai da eroare
        if self.right_panel:
            for widget in self.right_panel.winfo_children():
                widget.destroy()
        else:
            print("Eroare: right_panel nu exista!")
            return

        # 3. Indicator vizual ca lucreaza
        self.root.config(cursor="watch")
        self.root.update()

        try:
            # 4. Calcule
            if param_type == "energie" or param_type == "zcr":
                energie, rtz = self.proc.calcul_parametrii_timp_scurt()
                
                if len(energie) == 0: 
                    raise ValueError("Calculul a returnat date goale.")

                # FIX IMPORTANT: Salvam figura in self.analysis_fig
                self.analysis_fig = Figure(figsize=(6, 5), dpi=100) 
                ax = self.analysis_fig.add_subplot(111)

                if param_type == "energie":
                    pas = max(1, len(energie) // 3000)
                    ax.plot(energie[::pas], color='#2196F3', lw=1)
                    ax.set_title("Energie de Timp Scurt")
                    ax.set_ylabel("Magnitudine")
                else:
                    pas = max(1, len(rtz) // 3000)
                    ax.plot(rtz[::pas], color='#4CAF50', lw=1)
                    ax.set_title("Rata Trecerilor prin Zero (ZCR)")
                    ax.set_ylabel("Rată")
                
                ax.grid(True)

            elif param_type == "auto":
                auto = self.proc.calcul_autocorelatie()
                if len(auto) == 0:
                      raise ValueError("Autocorelatia a esuat.")
                
                # FIX IMPORTANT: Salvam figura in self.analysis_fig
                self.analysis_fig = Figure(figsize=(6, 5), dpi=100)
                ax = self.analysis_fig.add_subplot(111)
                
                pas = max(1, len(auto) // 3000)
                ax.plot(auto[::pas], color='#9C27B0', lw=1)
                ax.set_title("Funcția de Autocorelație")
                ax.set_ylabel("Magnitudine")
                ax.set_xlabel("Lag")
                ax.grid(True)

            # 5. Afisare Canvas - Folosim variabila self.analysis_fig
            self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, master=self.right_panel)
            self.analysis_canvas.draw()
            self.analysis_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # 6. Adaugare Toolbar
            toolbar_frame = tk.Frame(self.right_panel)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(self.analysis_canvas, toolbar_frame)
            toolbar.update()

        except Exception as e:
            tk.Label(self.right_panel, text=f"Eroare: {e}", fg="red", font=("Arial", 12)).pack(pady=50)
            print(f"Eroare Analiza: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.root.config(cursor="")

    # --- FUNCTIONALITATE GENERALA ---
    def on_tab_changed(self, event):
        self.stop_audio()
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 1 and self.proc.data is None:
             messagebox.showinfo("Info", "Încarcă un fișier audio în primul tab înainte de a face analiza.")
             self.notebook.select(0)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if path:
            self.stop_audio(); self.play_pos = 0 
            self.root.config(cursor="watch"); self.root.update()
            if self.proc.incarca_fisier(path):
                self.current_file_path = path; self.original_data_copy = self.proc.data.copy()
                self.playback_buffer = self.proc.data.astype(np.float32)
                self.root.title(f"Editor Audio - {path.split('/')[-1]}")
                self.line_wave = None; self.static_spec_line = None; self.plot_waveform()
                
                # Curatam panoul de analiza daca incarcam un fisier nou
                if self.right_panel and self.right_panel.winfo_exists():
                    for widget in self.right_panel.winfo_children(): widget.destroy()
                    self.lbl_analiza_info = tk.Label(self.right_panel, text="Selectează un parametru din stânga.", font=("Arial", 14), bg="white", fg="#555")
                    self.lbl_analiza_info.place(relx=0.5, rely=0.5, anchor="center")
            self.root.config(cursor="")

    def plot_waveform(self):
        if self.proc.data is None: return
        limit = 2000; step = max(1, len(self.proc.data) // limit)
        data_viz = self.proc.data[::step]; timp_viz = np.linspace(0, self.proc.duration, len(data_viz))
        if self.line_wave is None:
            self.ax_wave.clear(); self.line_wave, = self.ax_wave.plot(timp_viz, data_viz, color='#0055aa', lw=0.8)
            self.ax_wave.set_title(f"Durata: {self.proc.duration:.2f}s"); self.ax_wave.grid(True)
            self.cursor_line = self.ax_wave.axvline(x=0, color='red', lw=1.5, animated=True)
        else:
            self.line_wave.set_ydata(data_viz); self.line_wave.set_xdata(timp_viz)
            lim = max(abs(np.min(data_viz)), abs(np.max(data_viz))) if len(data_viz)>0 else 1
            self.ax_wave.set_ylim(-lim*1.1, lim*1.1)
        # Quick FFT
        try:
            fft_pts = 100000
            sl = self.proc.data if len(self.proc.data) < fft_pts else self.proc.data[len(self.proc.data)//2 - fft_pts//2 : len(self.proc.data)//2 + fft_pts//2]
            vals = np.abs(np.fft.rfft(sl)); freqs = np.fft.rfftfreq(len(sl), d=1./self.proc.fs)
            step_spec = max(1, len(freqs)//2000)
            if self.static_spec_line is None:
                self.ax_spec.clear(); self.static_spec_line, = self.ax_spec.plot(freqs[::step_spec], vals[::step_spec], color='#d32f2f', lw=0.8)
                self.ax_spec.grid(True); self.spec_line, = self.ax_spec.plot([], [], color='orange', animated=True)
            else:
                self.static_spec_line.set_ydata(vals[::step_spec]); self.static_spec_line.set_xdata(freqs[::step_spec]); self.ax_spec.set_ylim(0, max(vals)*1.1)
        except: pass
        self.canvas_main.draw()
        try: self.bg_wave = self.canvas_main.copy_from_bbox(self.ax_wave.bbox); self.bg_spec = self.canvas_main.copy_from_bbox(self.ax_spec.bbox)
        except: pass

    def play_audio(self):
        if not self.current_file_path or self.is_playing: return
        self.needs_stop = False
        def callback(outdata, frames, time, status):
            if self.playback_buffer is None: outdata.fill(0); return
            start = self.play_pos; end = start + frames
            if start >= len(self.playback_buffer): outdata.fill(0); self.needs_stop = True; return
            valid = min(len(self.playback_buffer) - start, frames)
            outdata[:valid, 0] = self.playback_buffer[start:start+valid]; self.play_pos += valid
            if valid < frames: outdata[valid:, 0] = 0
        self.stream = sd.OutputStream(samplerate=self.proc.fs, channels=1, callback=callback, blocksize=self.chunk_size)
        self.stream.start(); self.is_playing = True; self._schedule_gui_updates()

    def stop_audio(self):
        if self.stream: 
            try: self.stream.stop(); self.stream.close()
            except: pass
            self.stream = None
        self.is_playing = False; 
        if self._gui_after_id: self.root.after_cancel(self._gui_after_id); self._gui_after_id = None
        try: self.canvas_main.draw(); 
        except: pass

    def reset_cursor_to_start(self): self.stop_audio(); self.play_pos = 0; self.cursor_line.set_xdata([0]); self.canvas_main.draw()

    def _schedule_gui_updates(self):
        if self.needs_stop: self.stop_audio(); return
        cur_t = self.play_pos / self.proc.fs
        if self.bg_wave: self.canvas_main.restore_region(self.bg_wave)
        self.cursor_line.set_xdata([cur_t]); self.ax_wave.draw_artist(self.cursor_line); self.canvas_main.blit(self.ax_wave.bbox)
        fr, mg = self.proc.get_window_spectrum(self.play_pos, self.spk_win)
        if fr is not None and self.spec_line:
            if self.bg_spec: self.canvas_main.restore_region(self.bg_spec)
            self.spec_line.set_data(fr, np.log1p(mg)); self.ax_spec.draw_artist(self.spec_line); self.canvas_main.blit(self.ax_spec.bbox)
        self._gui_after_id = self.root.after(30, self._schedule_gui_updates)

    def _apply_filter(self, func, *args):
        if self.proc.data is None: return
        self.root.config(cursor="watch"); self.root.update()
        self.proc.data = self.original_data_copy.copy(); func(*args)
        self.playback_buffer = self.proc.data.astype(np.float32); self.plot_waveform(); self.root.config(cursor="")
    
    def apply_low_pass_from_slider(self): self._apply_filter(self.proc.aplica_filtru, 'low', self.scale_lp.get())
    def apply_high_pass_from_slider(self): self._apply_filter(self.proc.aplica_filtru, 'high', self.scale_hp.get())
    def apply_telephone(self): self._apply_filter(lambda: (self.proc.aplica_filtru('high', 300), self.proc.aplica_filtru('low', 3400)))
    def restore_clean_original(self): 
        if self.original_data_copy is not None: 
            self.proc.data = self.original_data_copy.copy(); self.playback_buffer = self.proc.data.astype(np.float32); self.plot_waveform()
    
    def on_closing(self): self.stop_audio(); self.root.destroy(); sys.exit(0)