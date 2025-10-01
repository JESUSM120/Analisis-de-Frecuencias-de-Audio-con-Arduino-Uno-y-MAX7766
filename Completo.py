import tkinter as tk
from tkinter import ttk, filedialog
import serial
import numpy as np
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.fft import fft
import threading
import queue

# --- CONFIGURACIÓN ---
SERIAL_PORT = 'COM6'  # ¡Asegúrate de que este puerto sea correcto!
BAUD_RATE = 115200
RECORDING_DURATION_SEC = 3.0  # Duración de la grabación en segundos
ANALYSIS_FREQ_CUTOFF_HZ = 1000 # <-- ¡NUEVO! Frecuencia máxima a mostrar en el eje X del análisis

# --- Variables Globales y de Sincronización ---
SAMPLES, SAMPLING_FREQUENCY = 0, 0
is_recording = False
is_analysis_window_open = False
stop_thread = threading.Event()
data_queue = queue.Queue()
analysis_queue = queue.Queue()
ser = None

# --- Conexión Serial (sin cambios) ---
print("Iniciando conexión serial...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Conectado a {SERIAL_PORT} a {BAUD_RATE} baudios.")
    print("Esperando configuración del Arduino...")
    for _ in range(10):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith('CONFIG:'):
                parts = line.replace('CONFIG:', '').split(',')
                SAMPLING_FREQUENCY = int(parts[0])
                SAMPLES = int(parts[1])
                print(f"Configuración recibida: Fs={SAMPLING_FREQUENCY} Hz, Muestras={SAMPLES}")
                break
except Exception as e:
    print(f"Error al conectar con el puerto serial: {e}")

if SAMPLING_FREQUENCY == 0:
    print("Error: No se recibió la configuración del Arduino. El programa no puede continuar.")
    if ser: ser.close()
    exit()

# --- Hilo de Lectura Serial (sin cambios) ---
def serial_reader_thread():
    global is_recording
    recorded_data = []
    
    while not stop_thread.is_set():
        try:
            if ser and ser.in_waiting > 0:
                serial_line = ser.readline().decode('utf-8').strip()

                if serial_line.startswith('FFT:'):
                    data = [float(val) for val in serial_line.replace('FFT:', '').split(',')]
                    if len(data) == (SAMPLES // 2) - 1:
                        data_queue.put(('FFT', data))

                elif serial_line.startswith('WAV:'):
                    data = [float(val) for val in serial_line.replace('WAV:', '').split(',')]
                    if len(data) == SAMPLES:
                        current_wave_block = np.array(data)
                        data_queue.put(('WAV', current_wave_block))
                        
                        if is_recording:
                            recorded_data.append(current_wave_block.copy())
                            blocks_per_second = SAMPLING_FREQUENCY / SAMPLES
                            num_blocks_needed = int(RECORDING_DURATION_SEC * blocks_per_second)
                            
                            if len(recorded_data) >= num_blocks_needed:
                                is_recording = False
                                full_signal = np.concatenate(recorded_data)
                                analysis_queue.put(full_signal)
                                recorded_data = []
        except Exception:
            time.sleep(0.1)
    print("Hilo de lectura detenido.")

# --- Funciones de Análisis y Almacenamiento ---
def start_recording():
    global is_recording
    if is_recording: return
    is_recording = True
    record_button.config(text="GRABANDO...", state=tk.DISABLED)
    print(f"Iniciando grabación de {RECORDING_DURATION_SEC} segundos...")

def calculate_fft(signal_data, fs):
    N = len(signal_data)
    signal_no_dc = signal_data - np.mean(signal_data)
    window = signal.windows.hamming(N)
    windowed_signal = signal_no_dc * window
    yf = fft(windowed_signal)
    magnitude_spectrum = np.abs(yf[:N//2]) * 2 / N
    return magnitude_spectrum[1:], np.fft.fftfreq(N, 1/fs)[:N//2][1:]

def open_analysis_window(original_wave, fs):
    global is_analysis_window_open
    is_analysis_window_open = True

    analysis_window = tk.Toplevel(root)
    analysis_window.title("Análisis Diferido de Grabación")
    analysis_window.geometry("1000x800")
    
    def _on_analysis_close():
        global is_analysis_window_open
        is_analysis_window_open = False
        analysis_window.destroy()
    
    analysis_window.protocol("WM_DELETE_WINDOW", _on_analysis_close)

    cutoff_freq = 1000
    b, a = signal.butter(4, cutoff_freq, btype='low', analog=False, fs=fs) 
    filtered_wave = signal.lfilter(b, a, original_wave)
    fft_orig_mag, fft_freq = calculate_fft(original_wave, fs)
    fft_filt_mag, _ = calculate_fft(filtered_wave, fs)

    fig = Figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221); ax1.plot(original_wave, 'gray'); ax1.set_title("Onda Original")
    ax2 = fig.add_subplot(222); ax2.plot(filtered_wave, 'b'); ax2.set_title("Onda Filtrada")
    
    # Gráfica de Espectro Original
    ax3 = fig.add_subplot(223)
    ax3.bar(fft_freq, fft_orig_mag, width=(fs / len(original_wave)), color='gray')
    ax3.set_title("Espectro Original")
    ax3.set_xlim(0, min(ANALYSIS_FREQ_CUTOFF_HZ, fs / 2)) # <-- MODIFICADO para usar el corte
    
    # Gráfica de Espectro Filtrado
    ax4 = fig.add_subplot(224)
    ax4.bar(fft_freq, fft_filt_mag, width=(fs / len(original_wave)), color='b')
    ax4.set_title("Espectro Filtrado")
    ax4.set_xlim(0, min(ANALYSIS_FREQ_CUTOFF_HZ, fs / 2)) # <-- MODIFICADO para usar el corte

    # Ajuste de Eje Y para ambos espectros
    max_spec = np.max(fft_orig_mag) if fft_orig_mag.size > 0 else 1
    ax3.set_ylim(0, max_spec * 1.1)
    ax4.set_ylim(0, max_spec * 1.1)
    
    fig.tight_layout()
    canvas_ana = FigureCanvasTkAgg(fig, master=analysis_window)
    canvas_ana.get_tk_widget().pack(fill=tk.BOTH, expand=1, padx=10, pady=5)
    
    ttk.Separator(analysis_window, orient='horizontal').pack(fill='x', padx=10, pady=5)
    button_frame = ttk.Frame(analysis_window)
    button_frame.pack(pady=10)
    ttk.Button(button_frame, text="Guardar Onda Original (.npy)", 
               command=lambda: save_data(original_wave, fs)).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Cargar Onda (.npy)", 
               command=load_and_analyze_data).pack(side=tk.LEFT, padx=10)

    analysis_window.focus_set()

def save_data(wave, fs):
    filepath = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
    if filepath: np.save(filepath, {'wave': wave, 'fs': fs}); print(f"Datos guardados en: {filepath}")

def load_and_analyze_data():
    filepath = filedialog.askopenfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
    if filepath:
        try:
            data = np.load(filepath, allow_pickle=True).item()
            open_analysis_window(data['wave'], data['fs'])
        except Exception as e: print(f"Error al cargar el archivo: {e}")

# --- CREACIÓN DE LA INTERFAZ GRÁFICA PRINCIPAL (sin cambios) ---
root = tk.Tk(); root.title("Analizador de Audio en Tiempo Real"); root.geometry("900x800")
control_frame = ttk.Frame(root); control_frame.pack(pady=10)
record_button = ttk.Button(control_frame, text="Grabar 3 Segundos", command=start_recording)
record_button.pack(side=tk.LEFT, padx=10)
ttk.Button(control_frame, text="Cargar Análisis", command=load_and_analyze_data).pack(side=tk.LEFT, padx=10)
main_plot_frame = ttk.Frame(root); main_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
fig_main = Figure(figsize=(8, 8))
ax_spec = fig_main.add_subplot(211)
x_freq = np.fft.rfftfreq(SAMPLES, 1/SAMPLING_FREQUENCY) if SAMPLING_FREQUENCY > 0 else []
bars = ax_spec.bar(x_freq[:SAMPLES//2], np.zeros(SAMPLES//2), width=(SAMPLING_FREQUENCY / SAMPLES if SAMPLES > 0 else 1) * 0.9)
ax_spec.set_ylim(0, 10); ax_spec.set_xlim(0, SAMPLING_FREQUENCY / 2); ax_spec.set_title("Espectro de Frecuencia (Tiempo Real)")
ax_wave = fig_main.add_subplot(212)
line, = ax_wave.plot(np.arange(0, SAMPLES), np.zeros(SAMPLES), color='r')
ax_wave.set_ylim(0, 1024); ax_wave.set_xlim(0, SAMPLES); ax_wave.set_title("Forma de Onda en el Tiempo (Tiempo Real)")
fig_main.tight_layout()
canvas_main = FigureCanvasTkAgg(fig_main, master=main_plot_frame)
canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# --- FUNCIÓN DE ACTUALIZACIÓN DE LA GUI (sin cambios) ---
def update_gui():
    if is_analysis_window_open:
        root.after(100, update_gui)
        return

    try:
        data_updated = False
        while not data_queue.empty():
            data_type, data = data_queue.get_nowait()
            data_updated = True
            if data_type == 'FFT':
                fft_data = np.zeros(SAMPLES // 2); fft_data[1:] = np.array(data)
                for bar, height in zip(bars, fft_data): bar.set_height(height)
                max_val = max(fft_data[1:]) if len(fft_data[1:]) > 0 else 10
                ax_spec.set_ylim(0, max(max_val * 1.1, 10))
            elif data_type == 'WAV':
                line.set_ydata(data)

        if data_updated:
            canvas_main.draw_idle()

        if not analysis_queue.empty():
            full_signal = analysis_queue.get_nowait()
            print(f"Grabación finalizada. Total de muestras: {len(full_signal)}")
            record_button.config(text="Grabar 3 Segundos", state=tk.NORMAL)
            open_analysis_window(full_signal, SAMPLING_FREQUENCY)
            
    except queue.Empty:
        pass
    
    root.after(50, update_gui)

# --- Cierre de la app (sin cambios) ---
def on_closing():
    print("\nCerrando la aplicación...")
    stop_thread.set()
    reader.join()
    if ser and ser.is_open: ser.close(); print("Puerto serial cerrado.")
    root.quit(); root.destroy()

# --- Iniciar la lógica de la aplicación (sin cambios) ---
root.protocol("WM_DELETE_WINDOW", on_closing)
reader = threading.Thread(target=serial_reader_thread, daemon=True)
reader.start()
root.after(100, update_gui)
root.mainloop()
print("Aplicación cerrada.")