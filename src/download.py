import os
import re
import warnings
from astropy.io import fits
import pandas as pd
from lightkurve import search_lightcurve
from multiprocessing import Process

# Suprime warnings
warnings.filterwarnings('ignore')

# Flag para retomar do último TIC baixado
RESUME_FROM_LAST = True

# Carrega e filtra labels
def load_labels(csv_path):
    df = pd.read_csv(csv_path, dtype={"tfopwg_disp": str})
    df = df.dropna(subset=["tfopwg_disp"])
    return df[df["tfopwg_disp"].isin({"CP", "FP", "KP", "FA"})]

# Sanitiza nomes para evitar caracteres problemáticos
def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", name)

# Identifica o último TIC baixado, baseado na hora de criação da pasta
def find_last_downloaded(base_output: str):
    last_time = 0
    last_label = None
    last_tid = None
    # percorre cada label e seus TICs
    for label in ["CP", "KP", "FP", "FA"]:
        label_dir = os.path.join(base_output, label)
        if not os.path.isdir(label_dir):
            continue
        for entry in os.listdir(label_dir):
            if entry.startswith('TIC_'):
                path = os.path.join(label_dir, entry)
                try:
                    ctime = os.path.getctime(path)
                except Exception:
                    continue
                if ctime > last_time:
                    last_time = ctime
                    last_label = label
                    last_tid = entry.replace('TIC_', '')
    return last_label, last_tid

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
                # Seleciona apenas a curva de maior duração por setor
                lcs = collection.download_all()
                sector_best = {}
                for lc in lcs:
                    if lc is None or not hasattr(lc, 'sector') or lc.time is None or len(lc.time) < 2:
                        continue
                    sector = lc.sector
                    duration = (lc.time[-1] - lc.time[0]).value
                    if sector not in sector_best or duration > sector_best[sector][1]:
                        sector_best[sector] = (lc, duration)

                # Salva apenas curvas selecionadas
                for sector, (lc, _) in sector_best.items():
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
    labels = {label: list(df[df["tfopwg_disp"] == label]["tid"].unique()) for label in ["CP", "KP", "FP", "FA"]}

    # Se for retomar, ajusta listas
    if RESUME_FROM_LAST:
        last_label, last_tid = find_last_downloaded(base_output)
        if last_label and last_tid:
            tids = labels.get(last_label, [])
            if last_tid in map(str, tids):
                idx = tids.index(int(last_tid))
                labels[last_label] = tids[idx:]
                print(f"[i] Retomando download em {last_label}, TIC {last_tid}")

    processes = []
    for label, tids in labels.items():
        if tids:
            p = Process(target=download_label, args=(label, tids, base_output))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "tic_labels.csv")
    download_all_spoc(CSV_PATH)