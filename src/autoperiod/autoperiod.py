# coding=utf-8
from __future__ import division, print_function, absolute_import

import math
import warnings

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import fftconvolve, savgol_filter
from scipy.stats import linregress

from plotting import Plotter


class Autoperiod:
    def __init__(self, times, values, threshold_method='mc', mc_iterations=100,
                 confidence_level=0.95, min_snr=3.0, min_phase_R=0.4, preprocessing_method='detrend_scale_smooth'):
        self.times = np.asarray(times)
        if self.times[0] != 0:
            self.times = self.times - self.times[0]
        self.original_values = np.asarray(values)
        self.values = self._preprocess(method=preprocessing_method)
        self._validate_inputs()

        self.min_snr = min_snr
        self.min_phase_R = min_phase_R
        self._threshold_method = threshold_method
        self.mc_iterations = mc_iterations
        self.confidence_level = confidence_level

        self.time_span = np.max(self.times) - np.min(self.times)
        intervals = np.diff(np.sort(self.times))
        self.min_interval = np.min(intervals) if len(intervals) > 0 else 0.0
        self.median_interval = np.median(intervals) if len(intervals) > 0 else 0.0

        self.freqs, self.powers = self._compute_periodogram()
        self.periods = 1 / self.freqs
        self._power_norm_factor = 1 / (2 * np.var(self.values - np.mean(self.values), ddof=1))

        self.acf, self.lags = self.autocorrelation(self.values)

        self._power_threshold = self._get_power_threshold()
        self._period_hints = self._get_period_hints()

        self._period = None
        self._validate_periods()

        self._phase_shift = 0.0
        self._sinwave = None
        if self._period:
            self._phase_shift = self._get_phase_shift(self._period)
            self._create_sinwave()

    @property
    def period(self):
        return self._period or np.nan

    @property
    def diagnostics(self):
        return {
            'candidates': [{
                'period': p,
                'metrics': self._calculate_all_metrics(p)
            } for p in self._period_hints],
            'selected_period': self._period,
            'thresholds': {
                'snr': self.min_snr,
                'phase_R': self.min_phase_R
            }
        }

    @property
    def normalized_powers(self):
        return self.powers * self._power_norm_factor

    @property
    def sinwave(self):
        return self._sinwave if self._period else None

    @property
    def max_period_threshold(self):
        return self.time_span / 2

    def period_blocks(self):
        if not self._period:
            return [], []

        period_region = self._sinwave > (np.max(self._sinwave) / 2)
        cutoff_indices = np.where(period_region[:-1] != period_region[1:])[0] + 1
        timeseries = np.vstack((self.times, self.values))
        blocks = np.array_split(timeseries, cutoff_indices, axis=1)

        on_blocks = blocks[::2] if period_region[0] else blocks[1::2]
        off_blocks = blocks[1::2] if period_region[0] else blocks[::2]
        return on_blocks, off_blocks

    def period_area(self):
        if not self._period:
            return 0, 0

        period_region = self._sinwave > (np.max(self._sinwave) / 2)
        on_area = np.trapz(self.values[period_region], self.times[period_region])
        off_area = np.trapz(self.values[~period_region], self.times[~period_region])
        return on_area, off_area

    def validate_hint(self, period):
        return all([
            self._check_phase_clustering(period),
            self._validate_acf_peak(period),
            self._local_snr(period) > 4.0
        ])

    def autocorrelation(self, data):
        if len(data) < 2:
            return np.array([]), np.array([])

        n = len(data)
        acf = fftconvolve(data, data[::-1], mode='full')[n - 1:]
        lags = np.arange(len(acf)) * self.median_interval
        return acf / np.max(acf), lags

    def _preprocess(self, method):
        processed = self.original_values.copy()

        if 'detrend' in method:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept = np.polyfit(self.times, processed, 1)
            processed -= slope * self.times + intercept

        if 'scale' in method:
            median = np.median(processed)
            mad = np.median(np.abs(processed - median))
            processed = (processed - median) / (1.4826 * mad + 1e-8)

        if 'smooth' in method and len(processed) >= 5:
            window = min(21, len(processed) // 3)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                processed = savgol_filter(processed, window, 2)

        return processed

    def _validate_inputs(self):
        if len(self.times) != len(self.values):
            raise ValueError("Times and values must have same length")
        if len(self.times) < 5:
            raise ValueError("At least 5 data points required")

    def _compute_periodogram(self):
        nyquist = 1 / (2 * self.min_interval) if self.min_interval > 0 else 1.0
        f_min = 2.0 / self.time_span if self.time_span > 0 else 0.01

        ls = LombScargle(self.times, self.values)
        freqs, powers = ls.autopower(
            minimum_frequency=f_min,
            maximum_frequency=nyquist,
            samples_per_peak=10,
            normalization='psd'
        )
        powers *= 2
        return freqs, powers

    def _validate_periods(self):
        for period in self._period_hints:
            if self._strict_validation(period):
                self._period = period
                return

        self._probabilistic_fallback()

    def _strict_validation(self, period):
        return all([
            self._validate_phase_clustering(period),
            self._validate_acf_peak(period),
            self._validate_snr(period)
        ])

    def _validate_phase_clustering(self, period):
        phase_info = self._calculate_phase_metrics(period)
        return (phase_info['p_value'] < 0.05) and (phase_info['R'] > self.min_phase_R)

    def _validate_snr(self, period):
        return self._calculate_snr(period) > self.min_snr

    def _validate_acf_peak(self, period):
        target_lag = int(round(period / self.median_interval))
        if target_lag >= len(self.acf):
            return False

        observed = self.acf[target_lag]
        n_boot = 500
        boot_peaks = []

        for _ in range(n_boot):
            boot_vals = self._block_bootstrap()
            boot_acf, _ = self.autocorrelation(boot_vals)
            if target_lag < len(boot_acf):
                boot_peaks.append(boot_acf[target_lag])

        if not boot_peaks:
            return False

        p_value = (np.sum(boot_peaks >= observed) + 1) / (len(boot_peaks) + 1)
        return p_value < 0.05

    def _probabilistic_fallback(self):
        candidates = []
        for period in self._period_hints:
            try:
                metrics = self._calculate_all_metrics(period)
                if metrics['phase']['R'] < self.min_phase_R:
                    continue
                score = self._calculate_composite_score(metrics)
                candidates.append((period, score, metrics))
            except Exception as e:
                continue

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            if best[1] > 0.5:
                self._period = best[0]
                warnings.warn(f"Selected period {best[0]:.2f} (Score: {best[1]:.2f}, R: {best[2]['phase']['R']:.2f})")


    def _get_power_threshold(self):
        if self._threshold_method == 'mc':
            return self._mc_threshold()
        elif self._threshold_method == 'stat':
            return self._stat_threshold()
        else:
            raise ValueError("Invalid threshold method")

    def _mc_threshold(self):
        max_powers = []
        for _ in range(self.mc_iterations):
            shuffled = np.random.permutation(self.values)
            _, powers = LombScargle(self.times, shuffled).autopower(
                normalization='psd',
                minimum_frequency=1 / self.time_span,
                maximum_frequency=1 / (2 * self.min_interval)
            )
            if len(powers) > 0:
                max_powers.append(np.max(powers))
        return np.percentile(max_powers, 99) * self._power_norm_factor if max_powers else 0

    def _stat_threshold(self):
        return -math.log(1 - (1 - self.confidence_level) ** (1 / len(self.powers)))

    def _get_period_hints(self):
        candidates = []
        for i, period in enumerate(self.periods):
            if (self.powers[i] * self._power_norm_factor > self._power_threshold and
                    2 * self.median_interval < period < self.time_span / 2):
                candidates.append((period, self.powers[i]))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in candidates[:5]]

    def _check_phase_clustering(self, period):
        folded = (self.times - self.times[0]) % period
        angles = 2 * np.pi * folded / period
        weights = (self.values - np.min(self.values)) / np.ptp(self.values)

        R = np.abs(np.sum(weights * np.exp(1j * angles))) / np.sum(weights)
        return R > 0.6

    def _block_bootstrap(self, block_size=5):
        n = len(self.values)
        n_blocks = int(np.ceil(n / block_size))

        indices = np.arange(0, n - block_size + 1, block_size)

        selected = np.random.choice(indices, size=n_blocks, replace=True)

        boot_sample = []
        for start in selected:
            end = min(start + block_size, n)
            boot_sample.append(self.values[start:end])

        return np.concatenate(boot_sample)[:n]

    def _local_snr(self, period):
        idx = np.abs(self.periods - period).argmin()
        window = slice(max(0, idx - 5), min(len(self.powers), idx + 6))
        noise_floor = np.median(self.powers[window])
        return (self.powers[idx] * self._power_norm_factor) / (noise_floor + 1e-8)

    def _get_phase_shift(self, period):
        phases = (self.times % period) / period
        weights = self.values - np.min(self.values)
        complex_phase = np.sum(weights * np.exp(2j * np.pi * phases))
        return period * np.angle(complex_phase) / (2 * np.pi)

    def _create_sinwave(self):
        t = self.times - self.times[0]
        amplitude = np.ptp(self.values) / 2
        self._sinwave = amplitude * np.cos(2 * np.pi / self._period * (t - self._phase_shift)) + np.median(self.values)

    def _calculate_all_metrics(self, period):
        return {
            'snr': self._calculate_snr(period),
            'phase': self._calculate_phase_metrics(period),
            'acf': self._calculate_acf_metrics(period)
        }

    def _calculate_composite_score(self, metrics):
        weights = {
            'snr': 0.4,
            'phase_R': 0.5,
            'acf': 0.1
        }

        snr_score = np.clip(metrics['snr'] / 50.0, 0, 1)
        phase_score = metrics['phase']['R']
        acf_score = metrics['acf']['confidence_ratio'] * 0.5

        return (
                weights['snr'] * snr_score +
                weights['phase_R'] * phase_score +
                weights['acf'] * acf_score
        )

    def _calculate_phase_metrics(self, period):
        folded = (self.times - self.times[0]) % period
        angles = 2 * np.pi * folded / period
        weights = (self.values - np.min(self.values)) / (np.ptp(self.values) + 1e-8)

        complex_sum = np.sum(weights * np.exp(1j * angles))
        R = np.abs(complex_sum) / np.sum(weights)
        z = (np.sum(weights) ** 2 * R ** 2) / np.sum(weights ** 2)
        p_value = np.exp(-z)

        return {
            'R': R,
            'p_value': p_value,
            'mean_phase': np.angle(complex_sum) % (2 * np.pi)
        }

    def _calculate_snr(self, period):
        idx = np.abs(self.periods - period).argmin()
        window_size = max(10, int(0.2 * len(self.powers)))
        start = max(0, idx - window_size)
        end = min(len(self.powers), idx + window_size)

        signal_power = self.powers[idx] * self._power_norm_factor
        noise_floor = np.mean(self.powers[np.r_[start:idx, idx + 1:end]]) * self._power_norm_factor
        return signal_power / (noise_floor + 1e-8)

    def _calculate_acf_metrics(self, period):
        target_lag = int(round(period / self.median_interval))
        if target_lag >= len(self.acf):
            return {'confidence_ratio': 0, 'p_value': 1}

        observed = self.acf[target_lag]
        n_boot = 500
        boot_peaks = []

        for _ in range(n_boot):
            boot_vals = self._block_bootstrap()
            boot_acf, _ = self.autocorrelation(boot_vals)
            if target_lag < len(boot_acf):
                boot_peaks.append(boot_acf[target_lag])

        if not boot_peaks:
            return {'confidence_ratio': 0, 'p_value': 1}

        ci_lower, ci_upper = np.percentile(boot_peaks, [5, 95])
        p_value = (np.sum(boot_peaks >= observed) + 1) / (len(boot_peaks) + 1)

        return {
            'confidence_ratio': observed / (ci_upper + 1e-8),
            'p_value': p_value
        }

    def _get_acf_search_range(self, period):
        period_idx = np.abs(self.periods - period).argmin()
        valid_indices = []

        if period_idx + 2 < len(self.periods):
            valid_indices.append(period_idx + 1)
            valid_indices.append(period_idx + 2)
        if period_idx - 2 >= 0:
            valid_indices.append(period_idx - 1)
            valid_indices.append(period_idx - 2)

        if not valid_indices:
            return 0, len(self.lags) - 1

        surrounding_periods = self.periods[valid_indices]
        min_period = np.min(surrounding_periods)
        max_period = np.max(surrounding_periods)

        min_lag = int(min_period / self.median_interval)
        max_lag = int(max_period / self.median_interval)
        return max(0, min_lag), min(len(self.lags) - 1, max_lag)

    def _check_acf_slope(self, search_min, search_max):
        min_err = float('inf')
        best_slopes = (0, 0)
        for t in range(search_min + 1, search_max):
            seg1_x = self.lags[search_min:t + 1]
            seg1_y = self.acf[search_min:t + 1]
            seg2_x = self.lags[t:search_max + 1]
            seg2_y = self.acf[t:search_max + 1]

            if len(seg1_x) < 3 or len(seg2_x) < 3:
                continue

            slope1, _, _, _, stderr1 = linregress(seg1_x, seg1_y)
            slope2, _, _, _, stderr2 = linregress(seg2_x, seg2_y)

            if stderr1 + stderr2 < min_err:
                min_err = stderr1 + stderr2
                best_slopes = (slope1, slope2)

        return best_slopes[0] > best_slopes[1] and not np.isclose(best_slopes[0], best_slopes[1], atol=0.01)



times = np.linspace(0, 40, 400)
values = 3 * np.sin(2 * np.pi * times / 20) + np.random.normal(0, 0.5, 400)

ap = Autoperiod(times, values, preprocessing_method='detrend_scale', min_phase_R=0.25)

plotter = Plotter(title="Star Light Curve Analysis")
plotter.plot_results(ap)
plotter.save("period_analysis.pdf")
plotter.show()
if ap.period:
    print(f"Detected period: {ap.period:.2f}")
    print("Validation metrics:", ap.diagnostics['candidates'][0]['metrics'])
else:
    print("No valid period detected")
    print("Top candidate metrics:", ap.diagnostics['candidates'][0]['metrics'])