import numpy as np
import pandas as pd
from lightkurve import LightCurve
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle

# -------------------------------------------------------------------------
# Funções de pré‐processamento
# -------------------------------------------------------------------------

def remove_outliers(lc: LightCurve, sigma=3) -> LightCurve:
    return lc.remove_outliers(sigma=sigma)


def detrend(lc: LightCurve, window_length=151, polyorder=2) -> LightCurve:
    flux = lc.flux.value
    smooth_flux = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
    detrended = flux - smooth_flux + np.nanmedian(flux)
    return LightCurve(time=lc.time, flux=detrended)


def normalize(lc: LightCurve) -> LightCurve:
    flux = lc.flux.value
    normed = flux / np.nanmedian(flux)
    return LightCurve(time=lc.time, flux=normed)


def interpolate_gaps(lc: LightCurve) -> LightCurve:
    flux = lc.flux.value
    series = pd.Series(flux).interpolate(method='linear', limit_direction='both')
    return LightCurve(time=lc.time, flux=series.to_numpy())


def estimate_period(lc: LightCurve, minimum_period=0.5, maximum_period=20) -> float:
    """
    Estima o período dominante via Lomb-Scargle e retorna em dias.
    """
    time = lc.time.value
    flux = lc.flux.value
    frequency, power = LombScargle(time, flux).autopower(
        minimum_frequency=1/maximum_period,
        maximum_frequency=1/minimum_period
    )
    best_freq = frequency[np.argmax(power)]
    return 1.0 / best_freq


def estimate_t0(lc: LightCurve, period: float) -> float:
    """
    Estima t0 como o instante de mínimo de fluxo na curva dobrada.
    """
    folded = lc.fold(period=period)
    idx_min = np.argmin(folded.flux.value)
    return folded.time.value[idx_min]


def phase_fold(lc: LightCurve, period: float, t0: float) -> LightCurve:
    return lc.fold(period=period, epoch_time=t0)


def resample_lightcurve(lc: LightCurve, num_points=256) -> np.ndarray:
    phase = lc.time.value
    flux = lc.flux.value
    idx = np.argsort(phase)
    phase = phase[idx]
    flux = flux[idx]
    _, unique = np.unique(phase, return_index=True)
    phase = phase[unique]
    flux = flux[unique]
    new_phase = np.linspace(phase.min(), phase.max(), num_points)
    f_interp = interp1d(phase, flux, kind='linear', fill_value='extrapolate')
    return new_phase, f_interp(new_phase)