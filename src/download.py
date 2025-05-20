import os
import re
import warnings
from astropy.io import fits
import pandas as pd
from lightkurve import search_lightcurve
from multiprocessing import Process

# Suprime warnings
warnings.filterwarnings('ignore')

# Carrega e filtra labels
def load_labels(csv_path):
    df = pd.read_csv(csv_path, dtype={"tfopwg_disp": str})
    df = df.dropna(subset=["tfopwg_disp"])
    return df[df["tfopwg_disp"].isin({"CP", "FP", "KP", "FA"})]

# Sanitiza nomes para evitar caracteres problemáticos
def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", name)

# Função de download para uma classe
def download_label(label: str, tids, base_output: str):
    label_dir = os.path.join(base_output, label)
    os.makedirs(label_dir, exist_ok=True)
    total = len(tids)
    print(f"[i] Iniciando classe {label}: {total} TICs")

    for idx, tid in enumerate(tids, start=1):
        tid_str = str(tid)
        tic_dir = os.path.join(label_dir, f"TIC_{sanitize(tid_str)}")
        os.makedirs(tic_dir, exist_ok=True)
        try:
            collection = search_lightcurve(f"TIC {tid_str}", mission="TESS", author="SPOC")
            if not collection:
                print(f"[!] Nenhuma curva para TIC {tid_str} em {label}")
            else:
                for lc in collection.download_all():
                    if lc is None:
                        continue
                    sector = lc.sector if hasattr(lc, 'sector') else 'NA'
                    fname = f"TIC_{sanitize(tid_str)}_sec{sector}.fits"
                    path = os.path.join(tic_dir, fname)

                    if os.path.isfile(path) and os.path.getsize(path) > 0:
                        print(f"[*] Já existe: {path}")
                        continue

                    try:
                        lc.to_fits(path, overwrite=True)
                        with fits.open(path): pass
                        print(f"[+] Salvo: {path}")
                    except Exception:
                        if os.path.isfile(path): os.remove(path)
                        try:
                            lc.to_fits(path, overwrite=True)
                            print(f"[+] Rebaixado: {path}")
                        except Exception as err:
                            print(f"[!] Falha ao baixar TIC {tid_str} setor {sector}: {err}")

            remaining = total - idx
            print(f"[i] Classe {label}: {remaining} TICs restantes")

        except Exception as e:
            print(f"[!] Erro geral {label} TIC {tid_str}: {e}")

    print(f"[i] Concluído classe {label}\n")

# Função principal: processos por label
def download_all_spoc(csv_path: str, base_output="../data/raw"):
    df = load_labels(csv_path)
    labels = {label: df[df["tfopwg_disp"] == label]["tid"].unique() for label in ["CP", "KP", "FP", "FA"]}
    processes = []
    for label, tids in labels.items():
        if len(tids) > 0:
            p = Process(target=download_label, args=(label, tids, base_output))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "tic_labels.csv")
    download_all_spoc(CSV_PATH)
