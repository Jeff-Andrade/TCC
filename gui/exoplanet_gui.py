# exoplanet_gui.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter.filedialog as fd

from src.features import extract_features_from_array


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

        ctk.CTkLabel(
            self.frame, text="🔭 Classificador de Exoplanetas",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=15)

        self.model_label = ctk.CTkLabel(self.frame, text="Nenhum modelo carregado", text_color="gray")
        self.model_label.pack()
        ctk.CTkButton(self.frame, text="Carregar Modelo", command=self.load_model).pack(pady=5)

        self.fits_label = ctk.CTkLabel(self.frame, text="Sem arquivos .fits carregados", text_color="gray")
        self.fits_label.pack()
        ctk.CTkButton(self.frame, text="Carregar .fits", command=self.load_fits).pack(pady=5)

        self.drop_frame = ctk.CTkFrame(self.frame, width=300, height=100, fg_color="#1a1a1a")
        self.drop_frame.pack(pady=10); self.drop_frame.pack_propagate(False)
        ctk.CTkLabel(self.drop_frame, text="Arraste e solte .fits aqui").pack(expand=True)
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self.drop_fits)

        self.classify_btn = ctk.CTkButton(
            self.frame, text="Classificar Todos",
            command=self.classify_all, state="disabled"
        )
        self.classify_btn.pack(pady=10)

        self.dropdown = ctk.CTkComboBox(self.frame, values=[], command=self.show_plot)
        self.dropdown.pack(pady=10); self.dropdown.set("Selecionar arquivo")

        self.output_label = ctk.CTkLabel(self.frame, text="", font=ctk.CTkFont(size=16))
        self.output_label.pack(pady=5)

        ctk.CTkButton(self.frame, text="Exibir Curva", command=self.show_plot).pack(pady=5)

    def load_model(self):
        path = fd.askopenfilename(filetypes=[("Joblib model", "*.joblib")])
        if not path: return
        try:
            self.model = joblib.load(path)
            self.model_label.configure(text="Modelo carregado ✅", text_color="green")
        except Exception as e:
            self.model_label.configure(text=f"Erro ao carregar: {e}", text_color="red")

    def load_fits(self):
        paths = fd.askopenfilenames(filetypes=[("FITS files", "*.fits")])
        self._add_fits(paths)

    def drop_fits(self, event):
        paths = [p.strip("{}") for p in event.data.split()]
        self._add_fits(paths)

    def _add_fits(self, paths):
        count = 0
        from astropy.io import fits
        for p in paths:
            try:
                with fits.open(p) as hdul:
                    data = hdul[1].data
                    time = data["TIME"]
                    flux = data["PDCSAP_FLUX"] if "PDCSAP_FLUX" in data.columns.names else data["FLUX"]
                mask = np.isfinite(time) & np.isfinite(flux)
                self.light_curves[p] = (time[mask], flux[mask])
                count += 1
            except:
                continue
        self.fits_label.configure(text=f"{count} arquivos prontos", text_color="green")
        if count:
            self.classify_btn.configure(state="normal")
            keys = list(self.light_curves.keys())
            self.dropdown.configure(values=keys)
            self.dropdown.set(keys[0])

    def classify_all(self):
        if not self.model:
            self.output_label.configure(text="⚠️ Carregue um modelo", text_color="orange")
            return
        results = []
        for fname, (time, flux) in self.light_curves.items():
            try:
                feat = extract_features_from_array(time, flux)
                if feat is None:
                    results.append(f"{os.path.basename(fname)} → Curva curta")
                    continue
                pred = self.model.predict(feat)[0]
                prob = self.model.predict_proba(feat)[0][pred]
                label = "Planeta" if pred == 1 else "Não-Planeta"
                results.append(f"{os.path.basename(fname)} → {label} ({prob*100:.1f}%)")
            except:
                results.append(f"{os.path.basename(fname)} → Erro")
        self.output_label.configure(text="\n".join(results[-10:]), text_color="lightblue")

    def show_plot(self, selected=None):
        key = selected or self.dropdown.get()
        if key in self.light_curves:
            time, flux = self.light_curves[key]
            plt.figure(figsize=(8,4))
            plt.plot(time, flux, ".", markersize=2)
            plt.xlabel("Time"); plt.ylabel("Flux")
            plt.title(os.path.basename(key))
            plt.tight_layout(); plt.show()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.title("Classificador de Exoplanetas")
    root.geometry("800x600")
    ctk.set_appearance_mode("Dark")
    app = ExoplanetClassifierApp(root)
    root.mainloop()
