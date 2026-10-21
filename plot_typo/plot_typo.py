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
    major_len: float = 8.0,
    minor_len: float = 4.0,
    major_width: float = 0.8,
    minor_width: float = 0.5,
) -> None:
    """
    Attiva tacche 'inout' (a cavallo dell'asse) primarie e secondarie
    su tutti i 4 lati, insieme a una griglia calibrata a doppio livello.
    """
    ax.minorticks_on()
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
    # Griglia tratteggiata morbida (stile report/paper accademici)
    ax.grid(True, which="major", ls="--", lw=0.6, color="#cccccc", alpha=0.85)
    ax.grid(True, which="minor", ls=":", lw=0.4, color="#e5e5e5", alpha=0.7)


if __name__ == "__main__":
    
    # Esempio 1
    
    # 1. Attiva lo stile Palatino LaTeX
    setup_academic_style()

    # 2. Definizione del modello e generazione dati
    A_true = 4.5e4
    mu_true = 105.7e3
    sigma_true = 4.2e3

    def gaussiana(x: np.ndarray) -> np.ndarray:
        return A_true * np.exp(-0.5 * ((x - mu_true) / sigma_true) ** 2)

    x_line = np.linspace(90e3, 120e3, 500)
    y_line = gaussiana(x_line)

    rng = np.random.default_rng(42)
    x_data = np.linspace(93e3, 118e3, 18)
    err_y = np.full_like(x_data, 0.05 * A_true)
    err_x = np.full_like(x_data, 250.0)
    y_data = gaussiana(x_data) + rng.normal(0, err_y)

    # 3. Costruzione esplicita del grafico
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    # Punti sperimentali
    ax.errorbar(
        x_data,
        y_data,
        xerr=err_x,
        yerr=err_y,
        fmt="o",
        markersize=3.2,
        color="black",
        ecolor="black",
        elinewidth=0.8,
        capsize=0,
        capthick=0.8,
        label=r"Conteggi rilevati",
        zorder=3,
    )

    # Curva teorica continua
    ax.plot(
        x_line,
        y_line,
        color="#1f77b4",
        lw=1.4,
        label=r"Curva Gaussiana teorica",
        zorder=2,
    )

    # Linea verticale di risonanza/centroide
    ax.axvline(
        mu_true,
        color="red",
        ls=":",
        lw=1.1,
        label=r"$\mu_0 = 105.7\,\mathrm{kHz}$",
        zorder=1,
    )

    # 4. Formattazione scientifica assi
    ax.set_xlabel(format_sci_axis(ax, x_data, r"Frequenza di scansione", axis="x", unit="Hz"))
    ax.set_ylabel(format_sci_axis(ax, y_line, r"Intensit\`a di conteggio", axis="y", unit="u.a."))

    # 5. Tacche e griglia accademica
    apply_academic_ticks_and_grid(ax)

    # 6. Legenda classica a box
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.4,
        handlelength=1.4,
    )

    fig.tight_layout(pad=0.3)

    # 7. Salvataggio
    save_folder = Path(__file__).resolve().parent / "examples"
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / "esempio_gaussiana.pdf"
    
    fig.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"Grafico salvato in: {out_file}")

    # Esempio 2
    
    import matplotlib.gridspec as gridspec
    
    x_curve = np.linspace(90e3, 122e3, 500)
    y_curve = gaussiana(x_curve)

    # Punti misurati con barre d'errore
    rng = np.random.default_rng(42)
    x_data = np.linspace(92e3, 120e3, 25)
    err_y = np.full_like(x_data, 0.04 * A_true)
    err_x = np.full_like(x_data, 200.0)
    y_data = gaussiana(x_data) + rng.normal(0, err_y)

    # Calcolo residui grezzi dei punti rispetto al modello
    y_pred_points = gaussiana(x_data)
    residui = y_data - y_pred_points

    # 3. Costruzione layout a 2 pannelli (hspace = 0)
    fig = plt.figure(figsize=(6.0, 5.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.0])
    gs.update(hspace=0.0)  # Incolla perfettamente i due grafici[cite: 1, 23]

    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

    # ── Pannello Superiore: Dati + Curva ──────────────────────────────────────
    ax_main.errorbar(
        x_data,
        y_data,
        xerr=err_x,
        yerr=err_y,
        fmt="o",
        markersize=3.0,
        color="black",
        ecolor="black",
        elinewidth=0.8,
        capsize=0,
        capthick=0.8,
        label=r"Dati sperimentali",
        zorder=3,
    )
    ax_main.plot(
        x_curve,
        y_curve,
        color="#1f77b4",  # Blu editoriale (o '#2ca02c' per verde)
        lw=1.4,
        label=r"Curva teorica",
        zorder=2,
    )
    ax_main.axvline(
        mu_true,
        color="red",
        ls=":",
        lw=1.1,
        label=r"$\mu_0 = 105.7\, \ \mathrm{kHz}$",
        zorder=1,
    )

    # ── Pannello Inferiore: Residui ───────────────────────────────────────────
    ax_res.errorbar(
        x_data,
        residui,
        yerr=err_y,
        fmt="o",
        markersize=3.0,
        color="black",
        ecolor="black",
        elinewidth=0.8,
        capsize=0,
        capthick=0.8,
        zorder=3,
    )
    ax_res.axhline(0.0, color="red", ls="--", lw=1.1, zorder=2)
    ax_res.axvline(mu_true, color="red", ls=":", lw=1.1, zorder=1)
    ax_res.set_yticks([-5e3, 0, 5e3])
    ax_res.set_ylim(-7.5e3, 7.5e3)

    # ── Formattazione assi con estrazione ordini di grandezza ─────────────────
    ax_main.set_ylabel(
        format_sci_axis(ax_main, y_curve, r"Intensit\`a", axis="y", unit="u.a.")
    )
    ax_res.set_ylabel(
        format_sci_axis(ax_res, residui, r"Residui", axis="y", unit="u.a.")
    )
    ax_res.set_xlabel(
        format_sci_axis(ax_res, x_data, r"Frequenza di scansione", axis="x", unit="Hz")
    )

    # ── Tacche 'inout' e griglie ──────────────────────────────────────────────
    for ax in [ax_main, ax_res]:
        apply_academic_ticks_and_grid(ax)

    # Nascondi numeri sull'asse x del grafico in alto per evitare sovrapposizioni
    ax_main.tick_params(axis="x", labelbottom=False)

    # Regolazione densità tick verticali per i residui
    ax_res.locator_params(axis="y", nbins=4)

    # ── Legenda rettangolare classica ─────────────────────────────────────────
    ax_main.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.4,
        handlelength=1.4,
    )

    fig.tight_layout(pad=0.25)

    # 4. Salvataggio PDF
    save_folder = Path(__file__).resolve().parent / "examples"
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / "esempio_scatter_residui.pdf"

    fig.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"Grafico salvato in: {out_file}")

    plt.show()