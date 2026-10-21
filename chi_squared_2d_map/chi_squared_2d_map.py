import math
from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy import ndimage
from iminuit import Minuit
import sys
from pathlib import Path
import matplotlib.patheffects as patheffects

typo_params = {
    "textsize" : 12,
    "labelsize" : 12,
    "ticksize" : 10,
    "legendsize" : 10,
    "titlesize" : 12,
    "width" : 10,
    "height" : 10
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino", "URW Palladio L", "TeX Gyre Pagella"],
    # Pacchetti LaTeX per formule e testo in Palatino coerente
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{mathpazo}       % Font Palatino classico per testo e math
        % In alternativa a mathpazo puoi usare i pacchetti moderni:
        % \usepackage{newpxtext}
        % \usepackage{newpxmath}
    """,
    # Dimensioni proporzionate del font
    "font.size": typo_params["textsize"],
    "axes.labelsize": typo_params["labelsize"],
    "xtick.labelsize": typo_params["ticksize"],
    "ytick.labelsize": typo_params["ticksize"],
    "legend.fontsize": typo_params["legendsize"],
    "figure.titlesize": typo_params["titlesize"],
})

# Importa FitPipeline direttamente dal tuo file chi_squared_fit.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PATH_FIT_MODULE = PROJECT_ROOT / "chi_squared_fit"

if str(PATH_FIT_MODULE) not in sys.path:
    sys.path.insert(0, str(PATH_FIT_MODULE))

try:
    from chi_squared_fit import FitPipeline, RisultatiFit
except ModuleNotFounddError:
    # Fallback se lo script viene eseguito direttamente dalla root di progetto
    from LABORATORIO_FISICA_CODICE.chi_squared_fit.chi_squared_fit import FitPipeline, RisultatiFit


# ──────────────────────────────────────────────────────────────────────────────
# Funzione per la generazione della Mappa Chi-Quadro Profilata (Metodo A)
# ──────────────────────────────────────────────────────────────────────────────
def chi2_map_pipeline(
    risultati: Any,
    param_x: str,
    param_y: str,
    param_info: Optional[Dict[str, Dict[str, str]]] = None,
    n_sigma: float = 3.5,
    grid_size: int = 35,
    zoom_factor: int = 4,
    show_clabels: bool = True,
    manual_clabels: Union[bool, List[Tuple[float, float]]] = False,
    clabel_fontsize: float = typo_params["legendsize"],
    stats_box_loc: str = "inside_map",  # 'inside_map', 'top', o 'custom'
    stats_box_pos: Optional[Tuple[float, float]] = None,
    show_stats_box: bool = True,
    spacing: float = 0.01,
    cmap: str = "viridis_r",
    save_path: Optional[str] = None,
    show: bool = True,
    colorbar_loc: str = "right",  # oppure "bottom"
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Genera la mappa 2D profilata del Delta chi^2 per la coppia (param_x, param_y)
    con profili 1D marginali e gestione elastica delle etichette.
    """
    m: Minuit = risultati.minuit_obj

    # 1. Limiti e deviazioni
    x_val = m.values[param_x]
    y_val = m.values[param_y]

    def _extract_err(param_name: str) -> float:
        if param_name in m.merrors:
            me = m.merrors[param_name]
            return max(abs(me.lower), abs(me.upper))
        return m.errors[param_name]

    err_x = _extract_err(param_x)
    err_y = _extract_err(param_y)

    x_min, x_max = x_val - n_sigma * err_x, x_val + n_sigma * err_x
    y_min, y_max = y_val - n_sigma * err_y, y_val + n_sigma * err_y
    global_fval = float(m.fval)

    # 2. Scansione profilata con Metodo A e Warm-Start sequenziale
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    X_map, Y_map = np.meshgrid(x_vals, y_vals)
    Z_profiled = np.zeros_like(X_map)

    orig_values = list(m.values)
    param_names = list(m.parameters)
    current_cache = {p: m.values[p] for p in param_names}

    for i in range(grid_size):
        j_indices = range(grid_size) if (i % 2 == 0) else reversed(range(grid_size))
        for j in j_indices:
            for p in param_names:
                m.values[p] = current_cache[p]

            m.fixed[param_x] = True
            m.fixed[param_y] = True
            m.values[param_x] = X_map[i, j]
            m.values[param_y] = Y_map[i, j]
            m.migrad()

            if not m.valid:
                for idx, p in enumerate(param_names):
                    m.values[p] = orig_values[idx]
                m.values[param_x] = X_map[i, j]
                m.values[param_y] = Y_map[i, j]
                m.migrad()

            Z_profiled[i, j] = m.fval
            for p in param_names:
                current_cache[p] = m.values[p]

    # Ripristino stato Minuit originale
    m.fixed[param_x] = False
    m.fixed[param_y] = False
    for idx, p in enumerate(param_names):
        m.values[p] = orig_values[idx]
    m.migrad()

    dz = Z_profiled - global_fval
    dz_smooth = ndimage.zoom(dz, zoom_factor, order=3)
    x_smooth = np.linspace(x_min, x_max, dz_smooth.shape[1])
    y_smooth = np.linspace(y_min, y_max, dz_smooth.shape[0])

    cbar_pad = 0.1

    if colorbar_loc == "right":
        fig = plt.figure(figsize=(8.2, 7.2))
        gs = gridspec.GridSpec(
            2, 4, 
            width_ratios=[3.5, 1.0, cbar_pad, 0.15], 
            height_ratios=[1.0, 3.5]
        )
        gs.update(wspace=spacing, hspace=spacing)

        ax1  = fig.add_subplot(gs[1, 0])
        ax_t = fig.add_subplot(gs[0, 0], sharex=ax1)
        ax_r = fig.add_subplot(gs[1, 1], sharey=ax1)
        cax  = fig.add_subplot(gs[1, 3])
        cbar_orient = "vertical"

    elif colorbar_loc == "bottom":
        fig = plt.figure(figsize=(7.5, 8.2))
        gs = gridspec.GridSpec(
            4, 2, 
            width_ratios=[3.5, 1.0], 
            height_ratios=[1.0, 3.5, cbar_pad, 0.15]
        )
        gs.update(wspace=spacing, hspace=spacing)
        
        ax1  = fig.add_subplot(gs[1, 0])
        ax_t = fig.add_subplot(gs[0, 0], sharex=ax1)
        ax_r = fig.add_subplot(gs[1, 1], sharey=ax1)
        cax  = fig.add_subplot(gs[3, 0])
        cbar_orient = "horizontal"

    else:
        fig = plt.figure(figsize=(7.5, 7.5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3.5, 1.0], height_ratios=[1.0, 3.5])
        gs.update(wspace=spacing, hspace=spacing)
        ax1  = fig.add_subplot(gs[1, 0])
        ax_t = fig.add_subplot(gs[0, 0], sharex=ax1)
        ax_r = fig.add_subplot(gs[1, 1], sharey=ax1)
        cax  = None

    # ── 4. Rendering Mappa con salvataggio oggetto ScalarMappable ─────────────
    im = ax1.pcolormesh(
        x_smooth, y_smooth, dz_smooth, 
        shading='gouraud', cmap=cmap, vmax=25, rasterized=True
    )

    # 4. Mappa 2D e contorni
    ax1.pcolormesh(x_smooth, y_smooth, dz_smooth, shading='gouraud', cmap=cmap, vmax=25, rasterized=True)

    cs_2d = ax1.contour(
        x_smooth, y_smooth, dz_smooth,
        levels=[2.30, 6.18, 11.83],
        colors=['white', 'white', 'white'],
        linewidths=[1.6, 1.2, 0.9],
        linestyles=['-', '--', ':']
    )
    cs_1d = ax1.contour(
        x_smooth, y_smooth, dz_smooth,
        levels=[1.00],
        colors=["#000000"],
        linewidths=[1.1],
        linestyles=['-.']
    )

    if show_clabels:
        labels_dict = {2.30: r'$1\sigma_{2D}$', 6.18: r'$2\sigma_{2D}$', 11.83: r'$3\sigma_{2D}$', 1.00: r'$1\sigma_{1D}$'}
        fmt_clabel = lambda v: labels_dict.get(round(v, 2), f"{v:.1f}")
        clabel_opts = dict(
                    inline=True,
                    fontsize=clabel_fontsize,
                    fmt=fmt_clabel,
                    manual=manual_clabels,
                    inline_spacing=5,
                    colors="black",
                )
        halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
        clbl_2d = ax1.clabel(cs_2d, **clabel_opts)
        plt.setp(clbl_2d, path_effects=halo)

        clbl_1d = ax1.clabel(cs_1d, **clabel_opts)
        plt.setp(clbl_1d, path_effects=halo)

    # 5. Profili 1D marginali Minos
    ax1.plot(x_val, y_val, marker='+', markersize=10, color='red', mew=2.0, zorder=5)

    for p_name, ax_proj, is_x in [(param_x, ax_t, True), (param_y, ax_r, False)]:
        if p_name in m.merrors:
            v_low = m.values[p_name] + m.merrors[p_name].lower
            v_up  = m.values[p_name] + m.merrors[p_name].upper
        else:
            v_low = m.values[p_name] - m.errors[p_name]
            v_up  = m.values[p_name] + m.errors[p_name]

        for v in [v_low, v_up]:
            if is_x:
                ax1.axvline(v, color="red", ls='--', lw=1.2, alpha=1, zorder=1)
                ax_t.axvline(v, color="red", ls='--', lw=1.2, alpha=1, zorder=1)
            else:
                ax1.axhline(v, color="red", ls='--', lw=1.2, alpha=1, zorder=1)
                ax_r.axhline(v, color="red", ls='--', lw=1.2, alpha=1, zorder=1)

    ax_t.axhline(1.0, color='green', ls='--', lw=1.2)
    ax_r.axvline(1.0, color='green', ls='--', lw=1.2)
    
    xp, yp = m.mnprofile(param_x, size=100, bound=(x_min, x_max))[:2]
    ax_t.plot(xp, yp - global_fval, color="black", lw=1.6)
    
    xp2, yp2 = m.mnprofile(param_y, size=100, bound=(y_min, y_max))[:2]
    ax_r.plot(yp2 - global_fval, xp2, color="black", lw=1.6)
    
    if cax is not None:
        cbar = fig.colorbar(im, cax=cax, orientation=cbar_orient)
        cbar.set_label(r'$\Delta\chi^2$', fontsize=typo_params["labelsize"])
        cbar.ax.tick_params(direction='inout', length=5, width=0.8)
        
        # Tacche discrete sulla colorbar coerenti con i livelli significativi
        cbar.set_ticks([0, 2.30, 6.18, 11.83, 20])
        cbar.set_ticklabels(['0', r'$1\sigma_{2D}$', r'$2\sigma_{2D}$', r'$3\sigma_{2D}$', '20'])

    # 6. Formattazione assi
    def _sci_label(ax, vals, nome, axis_letter):
        base = param_info[nome].get('label', nome) if (param_info and nome in param_info) else nome
        unit = param_info[nome].get('unit', '') if (param_info and nome in param_info) else ''
        amax = np.max(np.abs(vals))
        esp = int(math.floor(math.log10(amax))) if amax > 0 else 0
        scale = 10**esp
        suffix = (rf"\, [$\times 10^{{{esp}}}$ {unit}]" if esp != 0 else (f"\, [{unit}]" if unit else ""))
        fmt = ticker.FuncFormatter(lambda v, _: f"{v / scale:.2f}")
        getattr(ax, f"{axis_letter}axis").set_major_formatter(fmt)
        return f"{base}{suffix}"

    ax1.set_xlabel(_sci_label(ax1, [x_min, x_max], param_x, 'x'))
    ax1.set_ylabel(_sci_label(ax1, [y_min, y_max], param_y, 'y'))
    ax_t.set_ylabel(r"$\Delta\chi^2$")
    ax_r.set_xlabel(r"$\Delta\chi^2$")

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax_t.set_ylim(0, 5.5)
    ax_r.set_xlim(0, 5.5)

    ax1.tick_params(axis='both', which='both', labelbottom=True, labelleft=True, right=True, top=True)
    ax_t.tick_params(axis='x', bottom=True, labelbottom=False)
    ax_r.tick_params(axis='y', left=True, labelleft=False)
    ax_t.set_yticks([0, 1, 2, 4])
    ax_r.set_xticks([0, 1, 2, 4])

    for ax_m in [ax_t, ax_r]:
        ax_m.grid(True, ls=':', color='gray', alpha=0.5)
        
    for ax in [ax1, ax_t, ax_r]:
        ax.tick_params(
            axis='both',
            which='both',
            direction='inout', 
            length=8,          
            width=1,
            color='black'
        )

    # 7. Box Statistiche
    if show_stats_box:
        lx = param_info[param_x]['label'].replace('$', '') if (param_info and param_x in param_info) else param_x
        ly = param_info[param_y]['label'].replace('$', '') if (param_info and param_y in param_info) else param_y
        cov_val = m.covariance[param_x, param_y]
        corr_val = m.covariance.correlation()[param_x, param_y]

        def _fmt_sci(val: float) -> Tuple[str, int]:
            if val == 0:
                return '0', 0
            e = int(math.floor(math.log10(abs(val))))
            return f"{val / 10**e:.2g}", e

        cv_s, cv_e = _fmt_sci(cov_val)
        cov_tex = rf"{cv_s} \times 10^{{{cv_e}}}" if cv_e != 0 else cv_s
        box_kw = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=1, edgecolor='black', lw=0.6)

        if stats_box_loc == "inside_map":
            txt = rf"$\mathrm{{Cov}}({lx}, {ly}) = {cov_tex}$" + "\n" + rf"$\rho_{{{lx},{ly}}} = {corr_val:+.3f}$"
            ax1.text(0.96, 0.96, txt, transform=ax1.transAxes, fontsize=typo_params["textsize"], ha='right', va='top', bbox=box_kw, zorder=10)
        elif stats_box_loc == "top":
            txt = rf"$\mathrm{{Cov}}({lx}, {ly}) = {cov_tex}\quad\rho_{{{lx},{ly}}} = {corr_val:+.3f}$"
            ax_t.text(0.5, 1.15, txt, transform=ax_t.transAxes, fontsize=typo_params["textsize"], ha='center', va='bottom', bbox=box_kw)
        elif stats_box_loc == "custom" and stats_box_pos is not None:
            txt = rf"$\mathrm{{Cov}}({lx}, {ly}) = {cov_tex}$" + "\n" + rf"$\rho_{{{lx},{ly}}} = {corr_val:+.3f}$"
            fig.text(stats_box_pos[0], stats_box_pos[1], txt, fontsize=typo_params["textsize"], bbox=box_kw)

    fig.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mappa salvata con successo in: {save_path}")

    if show:
        plt.show()

    return fig, (ax1, ax_t, ax_r)


# ──────────────────────────────────────────────────────────────────────────────
# Test del flusso completo (Generazione Dati -> Fit ODR Metodo A -> Mappa 2D)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Definizione del modello fisico
    def modello_lineare(x, m, q):
        return m * x + q

    # 2. Generazione dati sintetici con errori su x e su y
    rng = np.random.default_rng(123)
    m_vero, q_vero = 2.45, 1.10
    n_punti = 35

    x_pulito = np.linspace(1.0, 10.0, n_punti)
    y_pulito = modello_lineare(x_pulito, m_vero, q_vero)

    err_x = np.full(n_punti, 0.20)
    err_y = np.full(n_punti, 0.45)

    x_misurato = x_pulito + rng.normal(0, err_x, n_punti)
    y_misurato = y_pulito + rng.normal(0, err_y, n_punti)

    # 3. Esecuzione FitPipeline con Metodo A (odr_lin_Newton=True)
    pipeline = FitPipeline(
        model=modello_lineare,
        param_names=["m", "q"],
        p0=[1.0, 0.0],
        run_minos=True,
        verbose=True,
    )

    risultati_fit = pipeline.fit(
        x=x_misurato,
        y=y_misurato,
        err_x=err_x,
        err_y=err_y,
        odr_lin_Newton=True,  # Attiva il Metodo A privo di bias a varianza congelata
    )

    # 4. Generazione Mappa Delta-Chi2 con layout flessibile
    info_parametri = {
        "m": {"label": r"$m$", "unit": r"V/s"},
        "q": {"label": r"$q$", "unit": r"V"},
    }

    current_dir = Path(__file__).resolve().parent
    save_folder = current_dir / "examples"
    save_folder.mkdir(parents=True, exist_ok=True)  # Crea la cartella se non esiste

    print("\nGenerazione mappa 2D in corso...")
    fig, axes = chi2_map_pipeline(
        risultati=risultati_fit,
        param_x="m",
        param_y="q",
        param_info=info_parametri,
        n_sigma=3.2,
        grid_size=30,
        stats_box_loc="custom",
        stats_box_pos= (0.725,0.85),
        show_clabels=True,
        manual_clabels=False,
        save_path=save_folder / "example_chi_squared_2d_map.pdf",
        show=True,
        cmap="viridis_r",
        spacing=0,
    )