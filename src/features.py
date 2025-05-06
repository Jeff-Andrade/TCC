# src/features.py

import os

import numpy as np
import pandas as pd
import pywt
from astropy.timeseries import LombScargle, BoxLeastSquares
from scipy.fft import rfft
from scipy.signal import savgol_filter, correlate
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Configurações
PROCESSED_DATA_DIR = "../data/processed"
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "tic_labels.csv")
FEATURES_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "features.csv")
VALID_LABELS = ["CP", "FP", "KP", "FA"]
MIN_LENGTH = 20  # mínimo de pontos após limpeza


def load_flux(tid, label):
    """Tenta carregar curva de luz processada (.npy)"""
    npy_path = os.path.join(PROCESSED_DATA_DIR, label, f"TIC_{tid}.npy")
    if os.path.exists(npy_path):
        flux = np.load(npy_path)
        time = np.linspace(0, 1, len(flux))
        return time, flux

    return None, None


def basic_stats(flux):
    if flux.size == 0 or not np.isfinite(flux).any():
        return (0,) * 6
    std_f = np.nanstd(flux)
    skew_f = skew(flux, nan_policy='omit')
    kurt_f = kurtosis(flux, nan_policy='omit')
    q25, q50, q75 = np.nanpercentile(flux, [25, 50, 75])
    return std_f, skew_f, kurt_f, q25, q50, q75


def fft_features(flux):
    if flux.size == 0:
        return 0, 0, 0
    fftv = np.abs(rfft(flux))
    total = np.sum(fftv) + 1e-8
    pratio = np.max(fftv) / total
    ten = np.sum(fftv ** 2)
    cent = entropy(fftv / total)
    return pratio, ten, cent


def lomb_scargle_features(time, flux):
    if time.size < 3 or flux.size < 3:
        return 0, 0
    ls = LombScargle(time, flux)
    freq, power = ls.autopower()
    if power.size == 0:
        return 0, 0
    peak_power = np.nanmax(power)
    peak_freq = freq[np.nanargmax(power)]
    rot_period = 1.0 / peak_freq if peak_freq > 0 else 0
    return peak_power, rot_period


def autocorr_features(flux):
    if flux.size == 0:
        return 0, 0, 0
    ac = correlate(flux, flux, mode='full')[flux.size - 1:]
    if len(ac) < 3:
        return 0, 0, 0
    peaks = sorted(np.argpartition(ac, -3)[-3:])
    return ac[peaks[0]], ac[peaks[1]], ac[peaks[2]]


def cwt_features(flux):
    if flux.size < 21:
        return 0, 0, 0
    coef, _ = pywt.cwt(flux, scales=np.arange(1, 21), wavelet='morl')
    energy = np.sum(coef ** 2, axis=1)
    return energy[2], energy[5], energy[10]


def bls_features(time, flux):
    if time.size < 3 or flux.size < 3:
        return (0,) * 5
    bls = BoxLeastSquares(time, flux)
    period_grid = np.linspace(0.5, 30, 10000)
    duration = 0.1
    results = bls.power(period_grid, duration)
    if results.power.size == 0:
        return (0,) * 5
    best = np.nanargmax(results.power)
    best_period = results.period[best]
    best_duration = results.duration[best]
    best_t0 = results.transit_time[best]
    best_depth = results.depth[best]
    best_power = results.power[best]
    snr = best_power / (np.median(results.power) + 1e-8)
    return best_period, best_depth, best_duration, snr, best_t0


def odd_even_depth(time, flux, period, dur, t0):
    if period <= 0 or dur <= 0:
        return 0
    depths = []
    epochs = int((time.max() - time.min()) / period)
    for i in range(epochs):
        seg_start = time.min() + i * period + t0 - dur / 2
        seg_end = seg_start + dur
        mask = (time >= seg_start) & (time <= seg_end)
        if np.sum(mask) > 5:
            depths.append(np.median(flux[mask]))
    if len(depths) > 2:
        return abs(np.nanmedian(depths[::2]) - np.nanmedian(depths[1::2]))
    return 0


def secondary_snr(flux, period, dur, t0, time):
    if period <= 0:
        return 0
    phase = (time - t0) % period
    sec_mask = (phase > (period / 2 - dur / 2)) & (phase < (period / 2 + dur / 2))
    if not np.any(sec_mask):
        return 0
    sec_depth = abs(np.nanmedian(flux[sec_mask]) - np.nanmedian(flux))
    return sec_depth / (np.nanstd(flux) + 1e-8)


def symmetry(flux):
    idx = np.argmin(flux)
    if 5 < idx < len(flux) - 5:
        L = flux[idx - 5:idx]
        R = flux[idx + 1:idx + 6][::-1]
        return -np.mean(np.abs(L - R))
    return 0


def binning(flux, n_bins=20):
    if flux.size < n_bins:
        return [0] * n_bins
    parts = np.array_split(flux, n_bins)
    return [np.nanmean(p) if len(p) > 0 else 0 for p in parts]


# Processamento principal
labels_df = pd.read_csv(PROCESSED_CSV_PATH)
features = []

for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extraindo features"):
    tid = int(row['tid'])
    label = row['tfopwg_disp']
    if label not in VALID_LABELS:
        continue

    time, flux = load_flux(tid, label)
    if flux is None or len(flux) < MIN_LENGTH:
        continue

    # Detrend e normalização
    detr = flux - savgol_filter(flux, 101, 3)
    flux_norm = MinMaxScaler().fit_transform(detr.reshape(-1, 1)).flatten()
    flux_norm = flux_norm[np.isfinite(flux_norm)]

    if len(flux_norm) < MIN_LENGTH:
        continue

    try:
        stats = basic_stats(flux_norm)
        fftf = fft_features(flux_norm)
        lsf = lomb_scargle_features(time, flux_norm)
        acs = autocorr_features(flux_norm)
        cwts = cwt_features(flux_norm)
        period, depth, dur, snr_val, t0 = bls_features(time, flux_norm)
        oe = odd_even_depth(time, flux_norm, period, dur, t0)
        ssnr = secondary_snr(flux_norm, period, dur, t0, time)
        sym = symmetry(flux_norm)
        bins = binning(flux_norm)

        feat = {
            'tid': tid, 'label': label,
            'period': period, 'depth': depth, 'duration': dur,
            'asymmetry': (acs[0] - acs[2]) / (acs[0] + acs[2] + 1e-8),
            'snr': snr_val,
            'std': stats[0], 'skew': stats[1], 'kurtosis': stats[2],
            'q25': stats[3], 'q50': stats[4], 'q75': stats[5],
            'pratio': fftf[0], 'fft_energy': fftf[1], 'entropy': fftf[2],
            'ls_peak_power': lsf[0], 'rotation_period': lsf[1],
            'ac1': acs[0], 'ac2': acs[1], 'ac3': acs[2],
            'cwt1': cwts[0], 'cwt2': cwts[1], 'cwt3': cwts[2],
            'odd_even_diff': oe, 'secondary_snr': ssnr, 'symmetry': sym
        }
        for i, v in enumerate(bins, 1):
            feat[f'bin_{i}'] = v
        features.append(feat)
    except Exception as e:
        print(f"Erro ao processar TIC {tid}: {e}")

# Salvar
if features:
    df_feat = pd.DataFrame(features)
    df_feat.to_csv(FEATURES_CSV_PATH, index=False)
    print(df_feat.head())
    print(f"\n✅ Features extraídas: {len(df_feat)} exemplos em {FEATURES_CSV_PATH}")
else:
    print("🚨 Nenhuma feature extraída. Verifique caminhos, rótulos e arquivos.")
