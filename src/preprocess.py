import os
import glob
import shutil
import numpy as np
import pandas as pd
from lightkurve import (
    read,
    LightCurve,
    LightCurveCollection,
    search_lightcurve
)
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count

from preprocessing import (
    remove_outliers,
    normalize,
    detrend,
    interpolate_gaps,
    resample_lightcurve
)

# --- Configurações ---
RAW_DATA_DIR       = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
RAW_CSV_PATH       = os.path.join(RAW_DATA_DIR, "tic_labels.csv")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "tic_labels.csv")
REPORT_PATH        = os.path.join(PROCESSED_DATA_DIR, "processing_report.txt")

N_POINTS        = 256
VALID_LABELS    = {"CP", "FP", "KP", "FA"}
MIN_USABLE_FRAC = 0.30   # mínimo 30% de cadências boas
MIN_MEDIAN_FLUX = 0      # mediana ≥ 0
TASK_TIMEOUT    = 300    # segundos por TIC antes de considerar timeout

def process_tic(record):
    """
    Processa um TIC inteiro: download, pré-processamento e salvamento.
    Retorna um dict com status, logs e razões de skip.
    """
    tid   = int(record["tid"])
    label = record["tfopwg_disp"]
    tic_dir = os.path.join(RAW_DATA_DIR, label, f"TIC_{tid}")
    status = {
        "tid": tid,
        "label": label,
        "processed": False,
        "downloaded": False,
        "failed_download": None,
        "corrupted": [],
        "skipped_reason": None,
        "log": []
    }

    os.makedirs(tic_dir, exist_ok=True)
    status["log"].append(f"→ [TIC {tid}] Diretório pronto: {tic_dir}")

    # 1) Procurar FITS locais
    fits_files = glob.glob(os.path.join(tic_dir, "*.fits"))
    status["log"].append(f"→ [TIC {tid}] Encontrados {len(fits_files)} FITS locais")

    # 2) Se não houver, baixar produtos SPOC
    if not fits_files:
        status["log"].append(f"→ [TIC {tid}] Iniciando download SPOC")
        try:
            sr = search_lightcurve(f"{tid}", mission="TESS", author="SPOC")
            if len(sr) == 0:
                status["skipped_reason"] = "nenhum produto SPOC encontrado"
                status["log"].append(f"✗ [TIC {tid}] Sem produtos SPOC")
                return status
            paths = sr.download_all(download_dir=tic_dir)
            status["downloaded"] = True
            status["log"].append(f"✓ [TIC {tid}] Download ({len(paths)} arquivos)")
            # mover e limpar subpastas
            for p in paths:
                src, dst = p.path, os.path.join(tic_dir, os.path.basename(p.path))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.move(src, dst)
            for root, dirs, _ in os.walk(tic_dir, topdown=False):
                for d in dirs:
                    full = os.path.join(root, d)
                    if not os.listdir(full):
                        os.rmdir(full)
        except Exception as e:
            status["failed_download"] = str(e)
            status["log"].append(f"✗ [TIC {tid}] Erro no download: {e}")
            return status

        fits_files = glob.glob(os.path.join(tic_dir, "*.fits"))
        status["log"].append(f"→ [TIC {tid}] Agora {len(fits_files)} FITS após download")
        if not fits_files:
            status["skipped_reason"] = "nenhum FITS após download"
            status["log"].append(f"✗ [TIC {tid}] Download não recuperou FITS")
            return status

    # 3) Ler FITS e aplicar máscara de qualidade
    total = good = 0
    lc_list = []
    for fn in fits_files:
        status["log"].append(f"→ [TIC {tid}] Lendo {os.path.basename(fn)}")
        try:
            part = read(fn)
        except Exception:
            status["corrupted"].append(fn)
            status["log"].append(f"✗ [TIC {tid}] Corrompido → removido")
            os.remove(fn)
            continue

        if hasattr(part, "quality"):
            q = np.asarray(part.quality)
            total += len(q)
            good  += np.sum(q == 0)
            mask = (q == 0)
            t = part.time.value[mask]
            f = part.flux.value[mask]
        else:
            t = part.time.value
            f = part.flux.value
            total += len(t)
            good  += len(t)

        lc_list.append(LightCurve(time=t, flux=f))
    status["log"].append(f"→ [TIC {tid}] Cadências totais={total}, boas={good}")

    # 4) Validar fração mínima e existência de segmentos
    if total == 0 or (good/total) < MIN_USABLE_FRAC:
        status["skipped_reason"] = f"baixa fração boas ({good/total:.1%})"
        status["log"].append(f"✗ [TIC {tid}] Skip qualidade insuficiente")
        return status
    if not lc_list:
        status["skipped_reason"] = "sem curvas válidas"
        status["log"].append(f"✗ [TIC {tid}] Skip sem segmentos válidos")
        return status

    # 5) Stitch (concatena e ordena por tempo)
    status["log"].append(f"→ [TIC {tid}] Stitching segmentos")
    stitched = LightCurveCollection(lc_list).stitch()
    lc = LightCurve(time=stitched.time.value, flux=stitched.flux.value)

    # 6) Checar mediana antes da normalização
    med = np.nanmedian(lc.flux)
    status["log"].append(f"→ [TIC {tid}] Mediana flux = {med:.2e}")
    if med < MIN_MEDIAN_FLUX:
        status["skipped_reason"] = f"mediana negativa ({med:.2e})"
        status["log"].append(f"✗ [TIC {tid}] Skip mediana negativa")
        return status

    # 7) Pipeline de pré-processamento
    status["log"].append(f"→ [TIC {tid}] Iniciando pipeline de pré-processamento")
    try:
        lc = remove_outliers(lc)
        lc = normalize(lc)
        lc = detrend(lc)
        lc = interpolate_gaps(lc)
        _, vec = resample_lightcurve(lc, num_points=N_POINTS)
    except Exception as e:
        status["skipped_reason"] = f"erro pipeline: {e}"
        status["log"].append(f"✗ [TIC {tid}] Pipeline error: {e}")
        return status

    # 8) Salvar vetor numpy
    out = os.path.join(PROCESSED_DATA_DIR, label, f"TIC_{tid}.npy")
    np.save(out, vec)
    status["processed"] = True
    status["log"].append(f"✓ [TIC {tid}] Salvo em {out}")
    return status

def main():
    # Preparar pastas
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for lbl in VALID_LABELS:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, lbl), exist_ok=True)

    # Ler CSV
    df = pd.read_csv(RAW_CSV_PATH)
    df = df[df["tfopwg_disp"].isin(VALID_LABELS)].reset_index(drop=True)
    records = df.to_dict(orient="records")
    total = len(records)

    print("=== Iniciando pipeline paralelo com timeout ===")
    print(f"Total de TICs: {total}")
    workers = max(1, cpu_count() - 3)
    print(f"Processos paralelos: {workers}, timeout por tarefa: {TASK_TIMEOUT}s")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        future_to_tid = {
            exe.submit(process_tic, rec): rec["tid"] for rec in records
        }
        for future in tqdm(as_completed(future_to_tid), total=total, desc="Processando"):
            tid = future_to_tid[future]
            try:
                res = future.result(timeout=TASK_TIMEOUT)
            except TimeoutError:
                print(f" [TIC {tid}] Timeout após {TASK_TIMEOUT}s — pulando")
                results.append({
                    "tid": tid, "label": None, "processed": False,
                    "downloaded": False, "failed_download": "timeout",
                    "corrupted": [], "skipped_reason": "timeout", "log": []
                })
            except Exception as e:
                print(f" [TIC {tid}] Erro inesperado: {e}")
                results.append({
                    "tid": tid, "label": None, "processed": False,
                    "downloaded": False, "failed_download": str(e),
                    "corrupted": [], "skipped_reason": str(e), "log": []
                })
            else:
                results.append(res)

    # Agregar resultados
    processed   = [r for r in results if r["processed"]]
    skipped     = [r for r in results if not r["processed"]]
    downloaded  = [r["tid"] for r in results if r.get("downloaded")]
    failed_dl   = [(r["tid"], r["failed_download"]) for r in results if r.get("failed_download")]
    corrupted   = [fn for r in results for fn in r["corrupted"]]

    # Salvar CSV final
    df_out = pd.DataFrame([
        {"tid": r["tid"], "tfopwg_disp": r["label"]}
        for r in processed
    ])
    df_out.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"→ CSV salvo: {PROCESSED_CSV_PATH}")

    # Gerar relatório
    with open(REPORT_PATH, "w") as rpt:
        rpt.write("=== Relatório de Processamento ===\n\n")
        rpt.write(f"Total de TICs: {total}\n")
        rpt.write(f"Processados com sucesso: {len(processed)}\n")
        rpt.write(f"Baixados: {len(downloaded)}\n")
        rpt.write(f"Timeouts/falhas download: {len(failed_dl)}\n")
        for tid, err in failed_dl:
            rpt.write(f"  - TIC {tid}: {err}\n")
        rpt.write(f"Corrompidos removidos: {len(corrupted)}\n")
        for fn in corrupted:
            rpt.write(f"  - {fn}\n")
        rpt.write(f"Pulados: {len(skipped)}\n")
        for r in skipped:
            rpt.write(f"  - TIC {r['tid']}: {r['skipped_reason']}\n")
        rpt.write("\nFim do relatório.\n")
    print(f"→ Relatório salvo: {REPORT_PATH}")

    print(" Pipeline concluído. Encerrando programa.")

if __name__ == "__main__":
    main()