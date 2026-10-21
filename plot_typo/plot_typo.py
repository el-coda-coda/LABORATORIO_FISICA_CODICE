"""
academic_plot.py
Modulo per la generazione di grafici scientifici publication-quality.
Include gestione automatica degli esponenti negli assi, tipografia Palatino LaTeX,
tacche 'inout' e griglie calibrate.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ──────────────────────────────────────────────────────────────────────────────
# 1. Configurazione Tipografica Globale (Palatino + LaTeX)
# ──────────────────────────────────────────────────────────────────────────────
def setup_academic_style(
    textsize: float = 11.0,
    labelsize: float = 11.5,
    ticksize: float = 10.0,
    legendsize: float = 9.5,
    titlesize: float = 11.5,
) -> None:
    """Configura i parametri rcParams di Matplotlib per l'estetica Palatino LaTeX."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino", "URW Palladio L", "TeX Gyre Pagella"],
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{mathpazo}
        """,
        "font.size": textsize,
        "axes.labelsize": labelsize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
        "legend.fontsize": legendsize,
        "figure.titlesize": titlesize,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 2. Funzioni Helper Fondamentali per Assi e Tacche
# ──────────────────────────────────────────────────────────────────────────────
def format_sci_axis(
    ax: plt.Axes,
    vals: Sequence[float] | np.ndarray,
    base_label: str,
    axis: str = "y",
    unit: str = "",
    min_exp: int = 2,
    dec: int = 2
) -> str:
    arr = np.asarray(vals, dtype=float)
    non_zero = arr[arr != 0]
    amax = np.max(np.abs(non_zero)) if len(non_zero) > 0 else 0.0
    esp = int(math.floor(math.log10(amax))) if amax > 0 else 0

    # Pulisce l'unità da eventuali \mathrm{} duplicati o dollari
    u_clean = unit.strip().replace("$", "")
    if u_clean.startswith(r"\mathrm{") and u_clean.endswith("}"):
        u_clean = u_clean[len(r"\mathrm{"):-1]

    # Unità racchiusa obbligatoriamente in math mode $\mathrm{...}$
    unit_math = rf"\mathrm{{{u_clean}}}" if u_clean else ""

    if abs(esp) >= min_exp:
        scale = 10.0**esp
        fmt = ticker.FuncFormatter(lambda v, _: f"{v / scale:.{dec}f}")
        getattr(ax, f"{axis}axis").set_major_formatter(fmt)

        if unit_math:
            suffix = rf"\, [$\times 10^{{{esp}}}\ {unit_math}$]"
        else:
            suffix = rf"\, [$\times 10^{{{esp}}}$]"
        return f"{base_label}{suffix}"

    suffix = rf"\, [${unit_math}$]" if unit_math else ""
    return f"{base_label}{suffix}"


def apply_academic_ticks_and_grid(
    ax: plt.Axes,
    scale_x: str = "linear",  # "linear" oppure "log"
    scale_y: str = "linear",  # "linear" oppure "log"
    major_len: float = 8.0,
    minor_len: float = 4.0,
    major_width: float = 0.8,
    minor_width: float = 0.5,
) -> None:
    """
    Configura tacche 'inout' primarie e secondarie su tutti i 4 lati
    e gestisce in modo indipendente griglie e formattazione per scale lineari o logaritmiche.
    """
    # 1. Impostazione scale
    ax.set_xscale(scale_x)
    ax.set_yscale(scale_y)

    # 2. Configurazione indipendente asse X
    if scale_x == "log":
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # 3. Configurazione indipendente asse Y
    if scale_y == "log":
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # 4. Parametri fisici delle tacche (centrate 'inout' su tutto il perimetro)
    ax.tick_params(
        axis="both",
        which="major",
        direction="inout",
        length=major_len,
        width=major_width,
        colors="black",
        top=True,
        right=True,
        bottom=True,
        left=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="inout",
        length=minor_len,
        width=minor_width,
        colors="black",
        top=True,
        right=True,
        bottom=True,
        left=True,
    )

    # 5. Griglia calibrata a doppio livello (tratteggiata major, punteggiata minor)
    ax.grid(True, which="major", ls="--", lw=0.6, color="#cccccc", alpha=0.85)
    ax.grid(True, which="minor", ls=":", lw=0.4, color="#e5e5e5", alpha=0.7)


if __name__ == "__main__":
    setup_academic_style()
    
    save_folder = Path(__file__).resolve().parent / "examples"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Modello comune per Esempio 1 ed Esempio 2
    A_true = 4.5e4
    mu_true = 105.7e3
    sigma_true = 4.2e3

    def gaussiana(x: np.ndarray) -> np.ndarray:
        return A_true * np.exp(-0.5 * ((x - mu_true) / sigma_true) ** 2)

    rng = np.random.default_rng(42)

    # ==========================================================================
    # Esempio 1: Scatter Gaussiana Singolo Pannello
    # ==========================================================================
    x_line = np.linspace(90e3, 120e3, 500)
    y_line = gaussiana(x_line)

    x_data = np.linspace(93e3, 118e3, 18)
    err_y = np.full_like(x_data, 0.05 * A_true)
    err_x = np.full_like(x_data, 250.0)
    y_data = gaussiana(x_data) + rng.normal(0, err_y)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    ax.errorbar(
        x_data, y_data, xerr=err_x, yerr=err_y,
        fmt="o", markersize=3.2, color="black", ecolor="black",
        elinewidth=0.8, capsize=0, capthick=0.8,
        label=r"Conteggi rilevati", zorder=3
    )
    ax.plot(x_line, y_line, color="#1f77b4", lw=1.4, label=r"Curva Gaussiana teorica", zorder=2)
    ax.axvline(mu_true, color="red", ls=":", lw=1.1, label=r"$\mu_0 = 105.7\,\mathrm{kHz}$", zorder=1)

    # 1. Configura prima tacche, scale e griglie
    apply_academic_ticks_and_grid(ax, scale_x="linear", scale_y="linear")

    # 2. Poi formatta le etichette e assegna il FuncFormatter
    ax.set_xlabel(format_sci_axis(ax, x_data, r"Frequenza di scansione", axis="x", unit="Hz"))
    ax.set_ylabel(format_sci_axis(ax, y_line, r"Intensit\`a di conteggio", axis="y", unit="u.a."))

    ax.legend(loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_folder / "example_1_gaussian_scatter.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    print("Grafico 1 salvato.")


    # ==========================================================================
    # Esempio 2: Gaussiana + Pannello Residui
    # ==========================================================================
    import matplotlib.gridspec as gridspec

    x_curve = np.linspace(90e3, 122e3, 500)
    y_curve = gaussiana(x_curve)

    x_data2 = np.linspace(92e3, 120e3, 25)
    err_y2 = np.full_like(x_data2, 0.04 * A_true)
    err_x2 = np.full_like(x_data2, 200.0)
    y_data2 = gaussiana(x_data2) + rng.normal(0, err_y2)
    residui = y_data2 - gaussiana(x_data2)

    fig = plt.figure(figsize=(6.0, 5.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.0])
    gs.update(hspace=0.0)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

    # Pannello superiore
    ax_main.errorbar(x_data2, y_data2, xerr=err_x2, yerr=err_y2, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=0, capthick=0.8, label=r"Dati sperimentali", zorder=3)
    ax_main.plot(x_curve, y_curve, color="#1f77b4", lw=1.4, label=r"Curva teorica", zorder=2)
    ax_main.axvline(mu_true, color="red", ls=":", lw=1.1, label=r"$\mu_0 = 105.7\,\mathrm{kHz}$", zorder=1)

    # Pannello inferiore
    ax_res.errorbar(x_data2, residui, yerr=err_y2, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=0, capthick=0.8, zorder=3)
    ax_res.axhline(0.0, color="red", ls="--", lw=1.1, zorder=2)
    ax_res.axvline(mu_true, color="red", ls=":", lw=1.1, zorder=1)

    # 1. Configura tacche e griglie prima di formattare
    apply_academic_ticks_and_grid(ax_main, scale_x="linear", scale_y="linear")
    apply_academic_ticks_and_grid(ax_res, scale_x="linear", scale_y="linear")
    ax_main.tick_params(axis="x", labelbottom=False)

    # 2. Impostazione range e locator per i residui (simmetrico e pulito)
    ax_res.set_yticks([-5e3, 0.0, 5e3])
    ax_res.set_ylim(-7.5e3, 7.5e3)
    ax_main.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="lower"))  # Toglie lo 0.00 che tocca la riga!

    # 3. Formattatori scientifici (chiamati per ULTIMI)
    ax_main.set_ylabel(format_sci_axis(ax_main, y_curve, r"Intensit\`a", axis="y", unit="u.a."))
    ax_res.set_ylabel(format_sci_axis(ax_res, residui, r"Residui", axis="y", unit="u.a.", dec=1))
    ax_res.set_xlabel(format_sci_axis(ax_res, x_data2, r"Frequenza di scansione", axis="x", unit="Hz"))

    ax_main.legend(loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
    fig.tight_layout(pad=0.25)
    fig.savefig(save_folder / "example_2_gaussian_residuals.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    print("Grafico 2 salvato.")


    # ==========================================================================
    # Esempio 3: Diagramma di Bode Modulo + Residui (Semilogaritmico)
    # ==========================================================================
    f0 = 10.0e3
    Q = 3.0

    def modulo_db(f: np.ndarray) -> np.ndarray:
        s = 1j * (f / f0)
        H = 1.0 / (1.0 + (s / Q) + s**2)
        return 20.0 * np.log10(np.abs(H))

    f_dense = np.logspace(3, 5, 500)
    mag_dense = modulo_db(f_dense)

    f_data3 = np.logspace(3.1, 4.9, 22)
    err_mag = np.full_like(f_data3, 0.45)
    mag_data = modulo_db(f_data3) + rng.normal(0, err_mag)
    residui_bode = mag_data - modulo_db(f_data3)

    fig = plt.figure(figsize=(6.0, 5.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.0])
    gs.update(hspace=0.0)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

    ax_main.plot(f_dense, mag_dense, color="#1f77b4", lw=1.4, label=r"Modello $|H(f)|$", zorder=2)
    ax_main.errorbar(f_data3, mag_data, yerr=err_mag, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=1.6, capthick=0.8, label=r"Dati misurati", zorder=3)
    ax_main.axvline(f0, color="red", ls=":", lw=1.1, label=r"$f_0 = 10\,\mathrm{kHz}$", zorder=1)

    ax_res.errorbar(f_data3, residui_bode, yerr=err_mag, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
    ax_res.axhline(0.0, color="red", ls="--", lw=1.1, zorder=2)
    ax_res.axvline(f0, color="red", ls=":", lw=1.1, zorder=1)

    apply_academic_ticks_and_grid(ax_main, scale_x="log", scale_y="linear")
    apply_academic_ticks_and_grid(ax_res, scale_x="log", scale_y="linear")
    ax_main.tick_params(axis="x", labelbottom=False)

    max_dev = np.max(np.abs(residui_bode) + err_mag)
    ax_res.set_ylim(-1.35 * max_dev, 1.35 * max_dev)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, symmetric=True, prune="both"))

    ax_main.set_ylabel(r"Ampiezza $[\mathrm{dB}]$")
    ax_res.set_xlabel(r"Frequenza $[\mathrm{Hz}]$")
    ax_res.set_ylabel(r"Residui $[\mathrm{dB}]$")

    ax_main.legend(loc="lower left", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
    fig.tight_layout(pad=0.25)
    fig.savefig(save_folder / "example_3_log_residuals.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    print("Grafico 3 salvato.")


    # ==========================================================================
    # Esempio 4: Grafico Log-Log Puro
    # ==========================================================================
    A_theo = 3.2e-4
    alpha = -1.65

    def power_law(f: np.ndarray) -> np.ndarray:
        return A_theo * (f**alpha)

    f_dense = np.logspace(1, 4, 400)
    S_dense = power_law(f_dense)

    f_data4 = np.logspace(1.2, 3.8, 16)
    S_true = power_law(f_data4)
    err_S = 0.12 * S_true
    S_data = S_true + rng.normal(0, err_S)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    ax.plot(f_dense, S_dense, color="#2ca02c", lw=1.4, label=rf"Modello $S(f) \propto f^{{{alpha:.2f}}}$", zorder=2)
    ax.errorbar(f_data4, S_data, yerr=err_S, fmt="o", markersize=3.2, color="black", ecolor="black", elinewidth=0.8, capsize=1.8, capthick=0.8, label=r"Dati spettrali", zorder=3)

    apply_academic_ticks_and_grid(ax, scale_x="log", scale_y="log")

    ax.set_xlabel(r"Frequenza $[\mathrm{Hz}]$")
    ax.set_ylabel(r"Densit\`a spettrale \, $[\mathrm{V^2 / Hz}]$")

    ax.legend(loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_folder / "example_4_scatter_log_both_axes.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    print("Grafico 4 salvato.")

    plt.show()