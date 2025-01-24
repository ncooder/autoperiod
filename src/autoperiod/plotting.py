import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, title="Autoperiod", figsize=(16, 12)):
        self.fig = plt.figure(figsize=figsize)
        self.title = title
        self._setup_axes()

    def _setup_axes(self):
        self.timeseries_ax = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        self.periodogram_ax = plt.subplot2grid((3, 3), (1, 0), colspan=3)
        self.acf_ax = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        self.phase_ax = plt.subplot2grid((3, 3), (2, 2))

        self.timeseries_ax.set(xlabel='Time', ylabel='Value', title='Time Series')
        self.periodogram_ax.set(xlabel='Period', ylabel='Power',
                                xscale='log', title='Periodogram')
        self.acf_ax.set(xlabel='Lag', ylabel='ACF', title='Autocorrelation')
        self.phase_ax.set(xlabel='Phase', ylabel='Value',
                          xticks=[0, 0.5, 1], title='Phase Folded')

    def _plot_acf(self, ap):
        self.acf_ax.clear()
        self.acf_ax.plot(ap.lags, ap.acf, 'b-', label='ACF')

        if ap.period:
            expected_lag = ap.period / ap.median_interval
            self.acf_ax.axvline(expected_lag, color='r', linestyle='--',
                                label=f'Expected Lag ({expected_lag:.1f})')

            peak_mask = ap.acf > np.quantile(ap.acf, 0.95)
            self.acf_ax.scatter(ap.lags[peak_mask], ap.acf[peak_mask],
                                color='orange', edgecolor='k', zorder=10,
                                label='Significant Peaks')

        self.acf_ax.legend()
        self.acf_ax.set_xlim(0, ap.lags[-1])

    def plot_results(self, autoperiod):

        self._plot_timeseries(autoperiod)

        self._plot_periodogram(autoperiod)

        self._plot_acf(autoperiod)

        if autoperiod.period:
            self._plot_phase_folded(autoperiod)

        plt.tight_layout()

    def _plot_timeseries(self, ap):
        self.timeseries_ax.plot(ap.times, ap.values, 'k.',
                                alpha=0.5, label='Data')
        if ap.period:
            self.timeseries_ax.plot(ap.times, ap.sinwave, 'r-',
                                    label=f'Periodic Model ({ap.period:.2f})')
        self.timeseries_ax.legend()

    def _plot_periodogram(self, ap):
        self.periodogram_ax.plot(ap.periods, ap.normalized_powers, 'b-')
        self.periodogram_ax.axhline(ap._power_threshold, color='r', linestyle='--',
                                    label=f'Threshold (p={ap.confidence_level})')
        self.periodogram_ax.axvline(ap.max_period_threshold, color='g', linestyle=':',
                                    label='Max Period Limit')

        if ap._period_hints:
            for p in ap._period_hints:
                self.periodogram_ax.axvline(p, color='orange', alpha=0.5)
        self.periodogram_ax.legend()

    def _plot_phase_folded(self, ap):
        if ap.sinwave is None:
            return
        phase = (ap.times % ap.period) / ap.period
        self.phase_ax.plot(phase, ap.values, 'k.', alpha=0.3)
        self.phase_ax.plot(np.sort(phase), ap.sinwave[np.argsort(phase)], 'r-')

    def save(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')

    def show(self):
        plt.show()