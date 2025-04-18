import os
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from sklearn.preprocessing import MinMaxScaler

from src.config import POS_DIR, NEG_DIR

def extract_features(tid, label):
    base_path = POS_DIR if label == 'planet' else NEG_DIR
    file_path = os.path.join(base_path, f"TIC_{tid}.fits")
    if not os.path.exists(file_path):
        return None
    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time = data['TIME']
            flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['FLUX']
            mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[mask], flux[mask]
            if len(time) < 10:
                return None

            flux = MinMaxScaler().fit_transform(flux.reshape(-1, 1)).flatten()

            std_flux = np.std(flux)
            skew_flux = skew(flux)
            kurt_flux = kurtosis(flux)

            fft_vals = np.abs(rfft(flux))
            fft_freqs = rfftfreq(len(flux), d=1)
            power_ratio = np.max(fft_vals) / np.sum(fft_vals)
            total_energy = np.sum(fft_vals ** 2)
            spectral_entropy = entropy(fft_vals / np.sum(fft_vals))

            try:
                bls = BoxLeastSquares(time, flux)
                periodogram = bls.autopower(0.2)
                period = periodogram.period[np.argmax(periodogram.power)]
                transit_depth = np.abs(np.min(flux) - np.median(flux))
            except:
                period, transit_depth = 0, 0

            mid_idx = np.argmin(flux)
            symmetry_score = 0
            if 5 < mid_idx < len(flux) - 5:
                left = flux[mid_idx - 5:mid_idx]
                right = flux[mid_idx + 1:mid_idx + 6][::-1]
                symmetry_score = -np.mean(np.abs(left - right))

            return [
                period, skew_flux, spectral_entropy,
                std_flux, total_energy, kurt_flux,
                power_ratio, symmetry_score, transit_depth
            ]
    except:
        return None

def extract_features_from_array(time, flux):
    if len(time) < 10 or len(flux) < 10:
        return None

    flux = MinMaxScaler().fit_transform(flux.reshape(-1, 1)).flatten()

    std_flux = np.std(flux)
    skew_flux = skew(flux)
    kurt_flux = kurtosis(flux)

    fft_vals = np.abs(rfft(flux))
    fft_freqs = rfftfreq(len(flux), d=1)
    power_ratio = np.max(fft_vals) / np.sum(fft_vals)
    total_energy = np.sum(fft_vals ** 2)
    spectral_entropy = entropy(fft_vals / np.sum(fft_vals))

    try:
        bls = BoxLeastSquares(time, flux)
        periodogram = bls.autopower(0.2)
        period = periodogram.period[np.argmax(periodogram.power)]
        transit_depth = np.abs(np.min(flux) - np.median(flux))
    except:
        period, transit_depth = 0, 0

    mid_idx = np.argmin(flux)
    symmetry_score = 0
    if 5 < mid_idx < len(flux) - 5:
        left = flux[mid_idx - 5:mid_idx]
        right = flux[mid_idx + 1:mid_idx + 6][::-1]
        symmetry_score = -np.mean(np.abs(left - right))

    return np.array([
        period, skew_flux, spectral_entropy,
        std_flux, total_energy, kurt_flux,
        power_ratio, symmetry_score, transit_depth
    ]).reshape(1, -1)