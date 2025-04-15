import tkinter.filedialog as fd

import joblib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

from src.feature_extractor import extract_features_from_array



class ExoplanetClassifierApp:
    def __init__(self, master):
        self.master = master
        self.model = None
        self.light_curves = {}
        self.predictions = {}

        self.setup_ui()

    def setup_ui(self):
        self.frame = ctk.CTkFrame(self.master)
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(self.frame, text="🔭 Classificador de Exoplanetas", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=15)

        self.model_label = ctk.CTkLabel(self.frame, text="Nenhum modelo carregado", text_color="gray")
        self.model_label.pack()
        ctk.CTkButton(self.frame, text="Carregar Modelo", command=self.load_model).pack(pady=5)

        self.fits_label = ctk.CTkLabel(self.frame, text="Sem arquivos .fits carregados", text_color="gray")
        self.fits_label.pack()
        ctk.CTkButton(self.frame, text="Carregar Arquivos .fits", command=self.load_fits).pack(pady=5)

        self.drop_frame = ctk.CTkFrame(self.frame, width=300, height=100, fg_color="#1a1a1a")
        self.drop_frame.pack(pady=10)
        self.drop_frame.pack_propagate(False)
        drop_label = ctk.CTkLabel(self.drop_frame, text="Arraste e Solte Arquivos .fits aqui")
        drop_label.pack(expand=True)

        # Bind drop
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self.drop_fits)

        self.classify_btn = ctk.CTkButton(self.frame, text="Classificar Todos", command=self.classify_all, state="disabled")
        self.classify_btn.pack(pady=10)

        self.dropdown = ctk.CTkComboBox(self.frame, values=[], command=self.update_plot)
        self.dropdown.pack(pady=10)
        self.dropdown.set("Caminhos de arquivo")

        self.output_label = ctk.CTkLabel(self.frame, text="", font=ctk.CTkFont(size=16))
        self.output_label.pack(pady=5)

        ctk.CTkButton(self.frame, text="Exibir Curva de Luz", command=self.show_plot).pack(pady=5)

    def load_model(self):
        path = fd.askopenfilename(filetypes=[("Modelo joblib", "*.joblib")])
        if path:
            try:
                self.model = joblib.load(path)
                self.model_label.configure(text="Modelo carregado ✅", text_color="green")
            except Exception as e:
                self.model_label.configure(text=f"Carregamento falhou: {e}", text_color="red")

    def load_fits(self):
        paths = fd.askopenfilenames(filetypes=[("Arquivos fits", "*.fits")])
        if paths:
            self.process_fits_files(paths)

    def drop_fits(self, event):
        paths = self.split_drop_paths(event.data)
        self.process_fits_files(paths)

    def split_drop_paths(self, data):
        return [p.strip('{}') for p in data.split()]

    def process_fits_files(self, paths):
        count = 0
        for path in paths:
            try:
                with fits.open(path) as hdul:
                    data = hdul[1].data
                    time = data['TIME']
                    flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['FLUX']
                    mask = np.isfinite(time) & np.isfinite(flux)
                    time, flux = time[mask], flux[mask]
                    if len(time) > 10:
                        self.light_curves[path] = (time, flux)
                        count += 1
            except Exception as e:
                print(f"Error loading {path}: {e}")
        self.fits_label.configure(text=f"{count} arquivos carregados", text_color="green")
        if count > 0:
            self.classify_btn.configure(state="normal")
            self.dropdown.configure(values=list(self.light_curves.keys()))
            self.dropdown.set(list(self.light_curves.keys())[0])

    def classify_all(self):
        if not self.model:
            self.output_label.configure(text="⚠️ Carregue um modelo primeiro!", text_color="orange")
            return

        results = []
        for fname, (time, flux) in self.light_curves.items():
            try:
                features = extract_features_from_array(time, flux)
                if features is not None:
                    pred = self.model.predict(features)[0]
                    prob = self.model.predict_proba(features)[0][pred]
                    label = "Planeta" if pred == 1 else "Não-Planeta"
                    self.predictions[fname] = (label, prob)
                    results.append(f"{fname.split('/')[-1]} → {label} ({prob*100:.1f}%)")
            except Exception as e:
                results.append(f"{fname} → Error: {e}")

        self.output_label.configure(text="\n".join(results[-3:]), text_color="lightblue")
        self.dropdown.configure(values=list(self.light_curves.keys()))
        self.dropdown.set(list(self.light_curves.keys())[0])

    def update_plot(self, selected_file):
        self.show_plot(selected_file)

    def show_plot(self, selected_file=None):
        if not selected_file:
            selected_file = self.dropdown.get()
        if selected_file in self.light_curves:
            time, flux = self.light_curves[selected_file]
            plt.style.use("dark_background")  # ✅ Valid dark style
            plt.figure(figsize=(10, 4))
            plt.plot(time, flux, color="cyan")
            plt.xlabel("Time")
            plt.ylabel("Flux")
            plt.title(selected_file.split("/")[-1])
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.title("Classificador de Exoplanetas")
    root.geometry("800x640")
    ctk.set_appearance_mode("Dark")
    app = ExoplanetClassifierApp(root)
    root.mainloop()
