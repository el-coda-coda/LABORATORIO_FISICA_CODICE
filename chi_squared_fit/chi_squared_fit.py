"""
Regressione automatica con ODR / curve_fit + iminuit + MINOS.

Flusso Metodo A (Standard di Ricerca):
  1. Stima iniziale locale con ODRPACK (se err_x > 0) o curve_fit (se err_x == 0).
  2. Fallback automatico con Dual Annealing se il chi2/dof locale supera la soglia di sicurezza.
  3. Costruzione della funzione di costo iminuit con varianza totale congelata sul minimo (Metodo A),
     oppure varianza mobile (Metodo Linearizzato) se odr_lin_Newton=False.
  4. Ottimizzazione finale con migrad(), calcolo errori simmetrici hesse() ed estrazione
     degli errori asimmetrici accurati con minos().
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from pathlib import Path
import numpy as np
from iminuit import Minuit
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit, dual_annealing, least_squares
from scipy.stats import chi2 as chi2_dist

# ──────────────────────────────────────────────────────────────────────────────
# Struttura dati di output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParametroFit:
    nome: str
    valore: float
    err_hesse: float
    err_minos_up: float = np.nan
    err_minos_down: float = np.nan
    minos_ok: bool = False
    asimmetria_assoluta: float = np.nan
    asimmetria_pct: float = np.nan

    def __str__(self) -> str:
        hesse = f"HESSE : ±{self.err_hesse:.3g}"
        if self.minos_ok:
            minos = (f"MINOS : +{self.err_minos_up:.3g} / -{self.err_minos_down:.3g}"
                     f"  |  asimmetria = {self.asimmetria_assoluta:+.3g}"
                     f"  ({self.asimmetria_pct:+.3f} %)")
        else:
            minos = "MINOS : non disponibile"
        return (f"  {self.nome:15} = {self.valore:.6e}\n"
                f"    {hesse}\n"
                f"    {minos}")

@dataclass
class RisultatiFit:
    parametri: list[ParametroFit]
    chi2: float
    chi2_rid: float
    dof: int
    p_value: float
    metodo_init: str
    valid: bool
    covariance: np.ndarray
    residui: np.ndarray          # Residui grezzi: y - f(x)
    pulls: np.ndarray            # Residui normalizzati: (y - f(x)) / sigma_tot
    sigma_tot: np.ndarray        # Incertezza efficace totale per ogni punto
    minuit_obj: Minuit

    def __getitem__(self, nome: str) -> ParametroFit:
        for p in self.parametri:
            if p.nome == nome:
                return p
        raise KeyError(nome)

    def valori(self) -> np.ndarray:
        return np.array([p.valore for p in self.parametri])

    def errori_hesse(self) -> np.ndarray:
        return np.array([p.err_hesse for p in self.parametri])

    def summary(self) -> str:
        cov_status = "OK" if (self.covariance is not None and not np.isnan(self.covariance).any()) else "FALLITA / NON DEFINITA"
        lines = [
            f"\n{'='*75}",
            f"  FitPipeline — risultati",
            f"  Init       : {self.metodo_init}",
            f"  Valid      : {self.valid}",
            f"  Covarianza : {cov_status}",
            f"  χ²         = {self.chi2:.4g}   χ²/dof = {self.chi2_rid:.4g}  ({self.dof} dof)   p-value = {self.p_value*1e2:.3f}%",
            f"{'='*75}",
        ]
        for p in self.parametri:
            lines.append(str(p))
        lines.append('='*75)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

# ──────────────────────────────────────────────────────────────────────────────
# Funzioni di costo per iminuit
# ──────────────────────────────────────────────────────────────────────────────

class _CostoODR_linearizzato:
    """Costo ODR a varianza effettiva mobile (soggetto a bias se i parametri sono nella derivata)."""
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, model: Callable, x: np.ndarray, y: np.ndarray, sigma_x: np.ndarray, sigma_y: np.ndarray):
        self._model = model
        self._x = x
        self._y = y
        self._sx = sigma_x
        self._sy = sigma_y
        self._use_sx = np.any(sigma_x > 0)

    def __call__(self, *params) -> float:
        y_pred = self._model(self._x, *params)
        if self._use_sx:
            x_scale = np.ptp(self._x) if np.ptp(self._x) > 0 else 1.0
            h = 1e-5 * (np.abs(self._x) + 0.01 * x_scale)
            dfdx = (self._model(self._x + h, *params) - self._model(self._x - h, *params)) / (2.0 * h)
            sigma2 = self._sy**2 + (dfdx * self._sx)**2
        else:
            sigma2 = self._sy**2
        return float(np.sum((self._y - y_pred)**2 / sigma2))


class _CostoODR:
    """Costo chi-quadro: Varianza congelata sul punto ottimo ODR (Privo di bias)."""
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, model: Callable, x: np.ndarray, y: np.ndarray, sigma_x: np.ndarray, sigma_y: np.ndarray, p_ref: Optional[np.ndarray] = None):
        self._model = model
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._sx = np.asarray(sigma_x, dtype=float)
        self._sy = np.asarray(sigma_y, dtype=float)
        self._use_sx = np.any(self._sx > 0)

        if self._use_sx:
            if p_ref is None:
                raise ValueError("Con errori su x (ODR), e' necessario fornire p_ref (parametri del pre-fit).")
            p_ref = np.asarray(p_ref, dtype=float)
            x_scale = np.ptp(self._x) if np.ptp(self._x) > 0 else 1.0
            h = 1e-5 * (np.abs(self._x) + 0.01 * x_scale)
            dfdx = (self._model(self._x + h, *p_ref) - self._model(self._x - h, *p_ref)) / (2.0 * h)
            self._sigma2 = self._sy**2 + (dfdx * self._sx)**2
        else:
            self._sigma2 = self._sy**2

    def __call__(self, *params) -> float:
        y_pred = self._model(self._x, *params)
        return float(np.sum((self._y - y_pred)**2 / self._sigma2))

# ──────────────────────────────────────────────────────────────────────────────
# Classe di fit principale
# ──────────────────────────────────────────────────────────────────────────────

class FitPipeline:
    def __init__(
        self,
        model: Callable,
        param_names: Sequence[str],
        p0: Sequence[float],
        limits: Optional[dict] = None,
        fixed: Optional[dict] = None,
        run_minos: bool = True,
        verbose: bool = True,
    ):
        self._model = model
        self._param_names = list(param_names)
        self._p0 = np.asarray(p0, dtype=float)
        self._limits = limits or {}
        self._fixed = fixed or {}
        self._run_minos = run_minos
        self._verbose = verbose

        if len(self._param_names) != len(self._p0):
            raise ValueError(f"param_names ({len(self._param_names)}) e p0 ({len(self._p0)}) devono avere la stessa lunghezza.")

    def _ricerca_globale_fallback(self, x: np.ndarray, y: np.ndarray, sx: np.ndarray, sy: np.ndarray, p_base: np.ndarray) -> np.ndarray:
        """Ricerca stocastica globale con Dual Annealing priva di assunzioni analitiche sul modello."""
        def objective(params):
            y_pred = self._model(x, *params)
            h = 1e-5 * (np.abs(x) + 0.01 * (np.ptp(x) if np.ptp(x) > 0 else 1.0))
            dfdx = (self._model(x + h, *params) - self._model(x - h, *params)) / (2.0 * h)
            sigma2 = sy**2 + (dfdx * sx)**2
            return float(np.sum(((y - y_pred)**2) / sigma2))

        bounds = []
        for name, val in zip(self._param_names, p_base):
            lim = self._limits.get(name, None)
            l_min = lim[0] if (lim and lim[0] is not None) else None
            l_max = lim[1] if (lim and lim[1] is not None) else None

            # Calcolo sicuro dei margini
            delta_val = 5.0 * abs(val) if val != 0 else 5.0
            low = l_min if l_min is not None else (val - delta_val)
            high = l_max if l_max is not None else (val + delta_val)

            # Garanzia bound consistente
            if low >= high:
                high = low + 1.0
            bounds.append((low, high))

        res_da = dual_annealing(objective, bounds=bounds, x0=p_base, maxiter=250, seed=42)
        return res_da.x

    def _stima_iniziale(self, x: np.ndarray, y: np.ndarray, err_x: np.ndarray, err_y: np.ndarray, use_odr: bool) -> tuple[np.ndarray, str]:
        def _odr_model(beta, t):
            return self._model(t, *beta)

        p_current = self._p0.copy()

        if use_odr:
            try:
                data = RealData(x, y, sx=err_x, sy=err_y)
                odr = ODR(data, Model(_odr_model), beta0=p_current)
                out = odr.run()
                dof = len(x) - len(p_current)
                chi2_rid = (out.sum_square / dof) if dof > 0 else 1.0

                if chi2_rid > 8.0:
                    warnings.warn(f"Fit ODR iniziale non ottimale (chi2/dof = {chi2_rid:.2f}). Avvio ricerca globale...", RuntimeWarning)
                    p_glob = self._ricerca_globale_fallback(x, y, err_x, err_y, p_current)
                    odr = ODR(data, Model(_odr_model), beta0=p_glob)
                    out = odr.run()
                    return out.beta, "ODR (Globale + Raffinamento)"

                return out.beta, "ODR"
            except Exception as exc:
                warnings.warn(f"ODR standard fallito ({exc}). Avvio fallback globale...")
                p_glob = self._ricerca_globale_fallback(x, y, err_x, err_y, p_current)
                odr = ODR(data, Model(_odr_model), beta0=p_glob)
                return odr.run().beta, "ODR (Globale Fallback)"
        else:
            try:
                popt, _ = curve_fit(self._model, x, y, p0=p_current, sigma=err_y, absolute_sigma=True)
                dof = len(x) - len(p_current)
                chi2_rid = np.sum(((y - self._model(x, *popt)) / err_y)**2) / dof
                if chi2_rid > 8.0:
                    p_glob = self._ricerca_globale_fallback(x, y, err_x, err_y, p_current)
                    popt, _ = curve_fit(self._model, x, y, p0=p_glob, sigma=err_y, absolute_sigma=True)
                return popt, "curve_fit"
            except Exception:
                p_glob = self._ricerca_globale_fallback(x, y, err_x, err_y, p_current)
                popt, _ = curve_fit(self._model, x, y, p0=p_glob, sigma=err_y, absolute_sigma=True)
                return popt, "curve_fit (Globale Fallback)"

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        err_x: Optional[np.ndarray] = None,
        err_y: Optional[np.ndarray] = None,
        odr_lin_Newton: bool = True, # Se True, utilizza il metodo ODR con varianza congelata, altrimenti utilizza il metodo linearizzato (soggetto a bias).
    ) -> RisultatiFit:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        err_x = np.zeros_like(x) if err_x is None else np.asarray(err_x, dtype=float)
        err_y = np.ones_like(y) if err_y is None else np.asarray(err_y, dtype=float)

        use_odr = np.any(err_x > 0)
        p_init, metodo = self._stima_iniziale(x, y, err_x, err_y, use_odr)

        if odr_lin_Newton:
            cost = _CostoODR(self._model, x, y, err_x, err_y, p_ref=p_init)
        else:
            cost = _CostoODR_linearizzato(self._model, x, y, err_x, err_y)

        m = Minuit(cost, *p_init, name=self._param_names)
        for nome, lim in self._limits.items():
            m.limits[nome] = lim
        for nome, fix in self._fixed.items():
            m.fixed[nome] = fix

        m.migrad()
        m.hesse()

        if not m.valid:
            warnings.warn("iminuit: il fit non è valido (m.valid = False).")

        return self._build_results(m, x, y, err_x, err_y, metodo)

    def _build_results(self, m: Minuit, x: np.ndarray, y: np.ndarray, err_x: np.ndarray, err_y: np.ndarray, metodo: str) -> RisultatiFit:
        parametri: list[ParametroFit] = []
        for nome in self._param_names:
            valore = m.values[nome]
            err_hesse = m.errors[nome]
            par = ParametroFit(nome=nome, valore=valore, err_hesse=err_hesse)

            if self._run_minos and not m.fixed[nome]:
                try:
                    m.minos(nome)
                    me = m.merrors[nome]
                    up, down = me.upper, -me.lower
                    media = (up + down) / 2.0
                    par.err_minos_up = up
                    par.err_minos_down = down
                    par.minos_ok = True
                    par.asimmetria_assoluta = up - down
                    par.asimmetria_pct = ((up - down) / media * 100.0) if media > 0 else float('nan')
                except Exception as exc:
                    warnings.warn(f"MINOS fallito per '{nome}': {exc}")

            parametri.append(par)

        y_pred = self._model(x, *m.values)
        residui = y - y_pred

        # Calcolo della varianza totale combinata ODR su ciascun punto
        if np.any(err_x > 0):
            x_scale = np.ptp(x) if np.ptp(x) > 0 else 1.0
            h = 1e-5 * (np.abs(x) + 0.01 * x_scale)
            dfdx = (self._model(x + h, *m.values) - self._model(x - h, *m.values)) / (2.0 * h)
            sigma_tot = np.sqrt(err_y**2 + (dfdx * err_x)**2)
        else:
            sigma_tot = err_y.copy()

        pulls = residui / sigma_tot
        dof = len(x) - m.nfit
        chi2_val = float(m.fval)
        chi2_rid = chi2_val / dof if dof > 0 else float('nan')
        p_val = chi2_dist.sf(chi2_val, dof) if dof > 0 else float('nan')

        cov_matrix = np.array(m.covariance) if m.covariance is not None else np.full((m.nfit, m.nfit), np.nan)

        res = RisultatiFit(
            parametri=parametri,
            chi2=chi2_val,
            chi2_rid=chi2_rid,
            dof=dof,
            p_value=p_val,
            metodo_init=metodo,
            valid=m.valid,
            covariance=cov_matrix,
            residui=residui,
            pulls=pulls,           
            sigma_tot=sigma_tot,
            minuit_obj=m,
        )

        if self._verbose:
            print(res.summary())

        return res

# ──────────────────────────────────────────────────────────────────────────────
# Funzione di costo GLOBALE (Puro iminuit)
# ──────────────────────────────────────────────────────────────────────────────

class _CostoGlobale:
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, models, xs, ys, err_xs, err_ys, p_ref: np.ndarray, odr_lin_Newton: bool = True):
        self.costi = []
        cost_cls = _CostoODR if odr_lin_Newton else _CostoODR_linearizzato
        for mod, x, y, sx, sy in zip(models, xs, ys, err_xs, err_ys):
            if odr_lin_Newton:
                self.costi.append(cost_cls(mod, x, y, sx, sy, p_ref=p_ref))
            else:
                self.costi.append(cost_cls(mod, x, y, sx, sy))

    def __call__(self, *params) -> float:
        return sum(costo(*params) for costo in self.costi)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Globale
# ──────────────────────────────────────────────────────────────────────────────

class GlobalFitPipeline:
    def __init__(
        self,
        models: Sequence[Callable],
        param_names: Sequence[str],
        p0: Sequence[float],
        limits: Optional[dict] = None,
        fixed: Optional[dict] = None,
        run_minos: bool = True,
        verbose: bool = True,
    ):
        self._models = models
        self._param_names = list(param_names)
        self._p0 = np.asarray(p0, dtype=float)
        self._limits = limits or {}
        self._fixed = fixed or {}
        self._run_minos = run_minos
        self._verbose = verbose

    def _stima_iniziale_globale(self, xs, ys, err_ys) -> tuple[np.ndarray, str]:
        def residui_obj(p):
            res = []
            for mod, x, y, sy in zip(self._models, xs, ys, err_ys):
                res.append((y - mod(x, *p)) / sy)
            return np.concatenate(res)

        bounds_lower = [-np.inf] * len(self._p0)
        bounds_upper = [ np.inf] * len(self._p0)
        for i, name in enumerate(self._param_names):
            if name in self._limits and self._limits[name] is not None:
                vmin, vmax = self._limits[name]
                bounds_lower[i] = vmin if vmin is not None else -np.inf
                bounds_upper[i] = vmax if vmax is not None else np.inf

        try:
            out = least_squares(residui_obj, x0=self._p0, bounds=(bounds_lower, bounds_upper), method='trf')
            return out.x, "least_squares_globale"
        except Exception:
            return self._p0.copy(), "p0_fallback"

    def fit(
        self,
        xs: Sequence[np.ndarray],
        ys: Sequence[np.ndarray],
        err_xs: Optional[Sequence[np.ndarray]] = None,
        err_ys: Optional[Sequence[np.ndarray]] = None,
        odr_lin_Newton: bool = True,
    ) -> RisultatiFit:
        xs = [np.asarray(x, dtype=float) for x in xs]
        ys = [np.asarray(y, dtype=float) for y in ys]
        err_xs = [np.zeros_like(x) for x in xs] if err_xs is None else [np.asarray(sx, dtype=float) for sx in err_xs]
        err_ys = [np.ones_like(y) for y in ys] if err_ys is None else [np.asarray(sy, dtype=float) for sy in err_ys]

        p_init, metodo_init = self._stima_iniziale_globale(xs, ys, err_ys)
        cost = _CostoGlobale(self._models, xs, ys, err_xs, err_ys, p_ref=p_init, odr_lin_Newton=odr_lin_Newton)

        m = Minuit(cost, *p_init, name=self._param_names)
        for nome, lim in self._limits.items():
            m.limits[nome] = lim
        for nome, fix in self._fixed.items():
            m.fixed[nome] = fix

        m.migrad()
        m.hesse()

        if not m.valid:
            warnings.warn("iminuit: il fit globale non è convergente (m.valid = False).")

        cov_matrix = np.array(m.covariance) if m.covariance is not None else np.full((m.nfit, m.nfit), np.nan)

        parametri = []
        for nome in self._param_names:
            par = ParametroFit(nome=nome, valore=m.values[nome], err_hesse=m.errors[nome])
            if self._run_minos and not m.fixed[nome]:
                try:
                    m.minos(nome)
                    me = m.merrors[nome]
                    par.err_minos_up = me.upper
                    par.err_minos_down = -me.lower
                    par.minos_ok = True
                    media = (par.err_minos_up + par.err_minos_down) / 2.0
                    par.asimmetria_assoluta = par.err_minos_up - par.err_minos_down
                    par.asimmetria_pct = (par.asimmetria_assoluta / media * 100.0) if media > 0 else np.nan
                except Exception as exc:
                    warnings.warn(f"MINOS fallito per '{nome}': {exc}")
            parametri.append(par)

        residui_list = [y - mod(x, *m.values) for mod, x, y in zip(self._models, xs, ys)]
        n_punti_tot = sum(len(x) for x in xs)
        dof = n_punti_tot - m.nfit
        chi2_val = float(m.fval)

        # Calcolo sigma_tot e pulls concatenati per la pipeline globale
        sigma_tot_list = []
        for mod, x, sx, sy in zip(self._models, xs, err_xs, err_ys):
            if np.any(sx > 0):
                x_scale = np.ptp(x) if np.ptp(x) > 0 else 1.0
                h = 1e-5 * (np.abs(x) + 0.01 * x_scale)
                dfdx = (mod(x + h, *m.values) - mod(x - h, *m.values)) / (2.0 * h)
                s_tot = np.sqrt(sy**2 + (dfdx * sx)**2)
            else:
                s_tot = sy.copy()
            sigma_tot_list.append(s_tot)

        residui_concat = np.concatenate(residui_list)
        sigma_tot_concat = np.concatenate(sigma_tot_list)
        pulls_concat = residui_concat / sigma_tot_concat

        res = RisultatiFit(
            parametri=parametri,
            chi2=chi2_val,
            chi2_rid=chi2_val / dof if dof > 0 else float('nan'),
            dof=dof,
            p_value=chi2_dist.sf(chi2_val, dof) if dof > 0 else float('nan'),
            metodo_init=f"iminuit (via {metodo_init})",
            valid=m.valid,
            covariance=cov_matrix,
            residui=residui_concat,
            pulls=pulls_concat,           
            sigma_tot=sigma_tot_concat,  
            minuit_obj=m,
        )

        if self._verbose:
            print(res.summary())

        return res

# ──────────────────────────────────────────────────────────────────────────────
# Esempi d'uso
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder = Path("examples")
    save_folder = folder if folder.is_dir() else Path(".")

    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 1: Oscillatore smorzato ad alta frequenza
    # ──────────────────────────────────────────────────────────────────────────────
    def smorzata(x, A, delta, omega, phi):
        return A * np.exp(-delta * x) * np.cos(omega * x + phi)

    rng = np.random.default_rng(42)
    t = np.linspace(0, 4e-5, 400)
    A_t, d_t, w_t, p_t = 1.5, 8e4, 5e5, 0.3
    y_t = smorzata(t, A_t, d_t, w_t, p_t)
    s_y = 0.05 * np.max(np.abs(y_t))
    s_x = 2e-8
    y_n = y_t + rng.normal(0, s_y, len(t))
    x_n = t + rng.normal(0, s_x, len(t))

    pip1 = FitPipeline(
        model=smorzata,
        param_names=["A", "delta", "omega", "phi"],
        p0=[1.0, 1e5, 4e5, 0.0],
        limits={"A": (0, None), "delta": (0, None)},
        run_minos=True,
        verbose=True,
    )

    res1 = pip1.fit(x_n, y_n, err_x=np.full_like(t, s_x), err_y=np.full_like(t, s_y))

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.errorbar(t, y_n, yerr=s_y, fmt='.', alpha=0.4, label='dati')
    ax1.plot(t, smorzata(t, *res1.valori()), 'r-', lw=1.5, label='fit')
    ax1.set_ylabel('y'); ax1.legend()
    ax2.errorbar(t, res1.residui, yerr=s_y, fmt='.', alpha=0.4)
    ax2.axhline(0, color='r', ls='--'); ax2.set_ylabel('residui'); ax2.set_xlabel('t [s]')
    fig1.tight_layout()
    fig1.savefig(f"{save_folder / 'fit_pipeline_esempio.pdf'}", dpi=300, bbox_inches="tight")
    print("\nGrafico Esempio 1 salvato.")

    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 2: Confronto ODR Newton vs ODR Linearizzato
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  ESEMPIO 2: Confronto Newton vs Linearizzato (Evidenza Bias)")
    print("=" * 75)

    def sinusoide(x, A, omega, phi):
        return A * np.sin(omega * x + phi)

    A_true, w_true, phi_true = 2.0, 14.0, 0.5
    N_pts = 120
    x_exact = np.linspace(0, 1.2, N_pts)
    y_exact = sinusoide(x_exact, A_true, w_true, phi_true)

    sigma_x_val = 0.015
    sigma_y_val = 0.15
    sx_arr = np.full(N_pts, sigma_x_val)
    sy_arr = np.full(N_pts, sigma_y_val)

    x_data = x_exact + rng.normal(0, sigma_x_val, N_pts)
    y_data = y_exact + rng.normal(0, sigma_y_val, N_pts)

    pip2 = FitPipeline(
        model=sinusoide,
        param_names=["A", "omega", "phi"],
        p0=[1.8, 11.0, 0.0],  # p0 volutamente distante per testare la ricerca globale!
        limits={"omega": (5.0, 25.0), "A": (0.5, 5.0)},
        run_minos=True,
        verbose=False,
    )

    # 1. Metodo A (ODR Newton congelato)
    res_newton = pip2.fit(x_data, y_data, err_x=sx_arr, err_y=sy_arr, odr_lin_Newton=True)

    # 2. Metodo Linearizzato (Effective Variance)
    res_lin = pip2.fit(x_data, y_data, err_x=sx_arr, err_y=sy_arr, odr_lin_Newton=False)

    print(f"{'Parametro':<10} | {'Valore Vero':<12} | {'ODR Newton':<15} | {'Linearizzato':<15} | {'Diff (Lin - New)':<15}")
    print("-" * 75)
    for p_name in ["A", "omega", "phi"]:
        v_true = {"A": A_true, "omega": w_true, "phi": phi_true}[p_name]
        v_new = res_newton[p_name].valore
        v_lin = res_lin[p_name].valore
        print(f"{p_name:<10} | {v_true:<12.4f} | {v_new:<15.4f} | {v_lin:<15.4f} | {v_lin - v_new:<+15.4f}")
    print("-" * 75)
    print(f"{'chi2 / dof':<10} | {'-':<12} | {res_newton.chi2_rid:<15.4f} | {res_lin.chi2_rid:<15.4f} | {res_lin.chi2_rid - res_newton.chi2_rid:<+15.4f}")
    print("=" * 75)

    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
    ax_top.errorbar(x_data, y_data, xerr=sx_arr, yerr=sy_arr, fmt='.', color='black', alpha=0.35, label='Dati con errori su x e y')
    ax_top.plot(x_plot, sinusoide(x_plot, *res_newton.valori()), 'b-', lw=2, label=f"ODR Newton (chi2/dof = {res_newton.chi2_rid:.2f})")
    ax_top.plot(x_plot, sinusoide(x_plot, *res_lin.valori()), 'r--', lw=1.5, label=f"Linearizzato (chi2/dof = {res_lin.chi2_rid:.2f})")
    ax_top.set_ylabel("y")
    ax_top.set_title("Confronto: ODR Newton vs ODR Linearizzato")
    ax_top.legend()

    ax_bot.plot(x_plot, sinusoide(x_plot, *res_lin.valori()) - sinusoide(x_plot, *res_newton.valori()), 'm-', lw=1.5, label='Diff curve (Lin - New)')
    ax_bot.axhline(0, color='gray', ls=':')
    ax_bot.set_ylabel("Delta Fit")
    ax_bot.set_xlabel("x")
    ax_bot.legend()
    fig2.tight_layout()
    fig2.savefig(f"{save_folder / 'fit_odr_difference.pdf'}", dpi=300, bbox_inches="tight")

    print("Grafico Esempio 2 salvato.")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 3: Retta di calibrazione con errori su x e y
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  ESEMPIO 3: Fit lineare (calibrazione strumentale)")
    print("=" * 75)

    def retta(x, m, q):
        return m * x + q

    m_true, q_true = 2.45, 1.10
    N_lin = 30
    x_lin_exact = np.linspace(1.0, 10.0, N_lin)
    y_lin_exact = retta(x_lin_exact, m_true, q_true)

    # Incertezze strumentali su x e y
    err_x_lin = np.full(N_lin, 0.25)
    err_y_lin = np.full(N_lin, 0.40)

    x_lin_meas = x_lin_exact + rng.normal(0, err_x_lin, N_lin)
    y_lin_meas = y_lin_exact + rng.normal(0, err_y_lin, N_lin)

    pip_lin = FitPipeline(
        model=retta,
        param_names=["m", "q"],
        p0=[1.0, 0.0],
        run_minos=True,
        verbose=True,
    )

    res_lin = pip_lin.fit(x_lin_meas, y_lin_meas, err_x=err_x_lin, err_y=err_y_lin)

    # Grafico con bande di errore bidimensionali
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.errorbar(x_lin_meas, y_lin_meas, xerr=err_x_lin, yerr=err_y_lin, fmt='o', color='black', alpha=0.6, label='Dati sperimentali')
    x_line_plot = np.linspace(0.5, 10.5, 200)
    ax1.plot(x_line_plot, retta(x_line_plot, *res_lin.valori()), 'b-', lw=1.5, label='Fit ODR Metodo A')
    ax1.set_ylabel("Segnale misurato $y$ [V]")
    ax1.set_title("Calibrazione lineare con incertezze su $x$ e $y$")
    ax1.legend()

    # Residui normalizzati (pulls)
    # Calcolo dell'errore totale efficace per i residui
    m_fit = res_lin["m"].valore
    ax2.errorbar(x_lin_meas, res_lin.pulls, yerr=np.ones_like(x_lin_meas), fmt='o', color='black', alpha=0.6)
    ax2.axhline(0, color='r', ls='--')
    ax2.set_ylabel("Pulls [$\\sigma$]")
    ax2.set_xlabel("Grandezza applicata $x$ [u.a.]")
    ax2.set_xlabel("Grandezza applicata $x$ [u.a.]")
    fig3.tight_layout()
    fig3.savefig(f"{save_folder / 'esempio_3_fit_lineare.pdf'}", dpi=300, bbox_inches="tight")

    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 4: Picco Gaussiano con Fondo
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  ESEMPIO 4: Picco Gaussiano spettroscopico su fondo piatto")
    print("=" * 75)

    def gaussiana_fondo(x, N, mu, sigma, bg):
        return (N / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

    N_true, mu_true, sigma_true, bg_true = 50.0, 5.0, 0.6, 2.0
    N_pts_gauss = 80
    x_g = np.linspace(1.0, 9.0, N_pts_gauss)
    y_g_clean = gaussiana_fondo(x_g, N_true, mu_true, sigma_true, bg_true)

    err_x_g = np.full(N_pts_gauss, 0.04)
    # Errore su y dipendente dal conteggio (Poisson approssimato)
    err_y_g = np.sqrt(y_g_clean) * 0.4 + 0.1

    x_g_meas = x_g + rng.normal(0, err_x_g, N_pts_gauss)
    y_g_meas = y_g_clean + rng.normal(0, err_y_g, N_pts_gauss)

    pip_gauss = FitPipeline(
        model=gaussiana_fondo,
        param_names=["N", "mu", "sigma", "bg"],
        p0=[30.0, 4.8, 0.8, 1.5],
        limits={"N": (0.0, None), "sigma": (0.05, 3.0), "bg": (0.0, None)},
        run_minos=True,
        verbose=True,
    )

    res_gauss = pip_gauss.fit(x_g_meas, y_g_meas, err_x=err_x_g, err_y=err_y_g)

    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.errorbar(x_g_meas, y_g_meas, xerr=err_x_g, yerr=err_y_g, fmt='.', color='black', alpha=0.5, label='Conteggi')
    x_dense = np.linspace(1.0, 9.0, 300)
    ax1.plot(x_dense, gaussiana_fondo(x_dense, *res_gauss.valori()), 'g-', lw=1.8, label='Fit ODR Gaussiana')
    ax1.axhline(res_gauss["bg"].valore, color='gray', ls=':', label='Fondo stimato')
    ax1.set_ylabel("Intensità / Conteggi")
    ax1.set_title("Spettro con picco gaussiano e risoluzione finita su $x$")
    ax1.legend()

    # Pannello residui normalizzati (Pulls standardizzati attorno a 0 con sigma = 1)
    ax2.errorbar(x_g_meas, res_gauss.pulls, yerr=np.ones_like(x_g_meas), fmt='.', alpha=0.5)
    ax2.axhline(0, color='r', ls='--')
    ax2.axhline(+1, color='gray', ls=':', alpha=0.7)
    ax2.axhline(-1, color='gray', ls=':', alpha=0.7)
    ax2.set_ylabel("Pulls [$\\sigma$]")
    ax2.set_xlabel("Canale energetico $x$ [keV]")
    fig4.tight_layout()
    fig4.savefig(f"{save_folder / 'esempio_4_gaussiana.pdf'}", dpi=300, bbox_inches="tight")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 5: Stress-Test e recupero automatico da p0 sballati
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  ESEMPIO 5: Stress-Test con p0 completamente errati")
    print("=" * 75)

    def oscillatore_anharmonico(x, A, w, offset):
        return A * np.cos(w * x) + offset

    N_stress = 100
    x_st = np.linspace(0.0, 3.0, N_stress)
    A_real, w_real, off_real = 3.0, 8.0, 1.5
    y_st = oscillatore_anharmonico(x_st, A_real, w_real, off_real) + rng.normal(0, 0.2, N_stress)
    sx_st = np.full(N_stress, 0.02)
    sy_st = np.full(N_stress, 0.2)

    # Parametri iniziali p0 volutamente disastrosi (frequenza tripla e ampiezza errata)
    p0_sballato = [0.5, 23.0, -1.0]

    print(f"Valori veri generati : A = {A_real}, w = {w_real}, offset = {off_real}")
    print(f"Guess fornito (p0)   : A = {p0_sballato[0]}, w = {p0_sballato[1]}, offset = {p0_sballato[2]}")
    print("Esecuzione pipeline (ci si attende l'avviso di recupero globale)...")

    pip_stress = FitPipeline(
        model=oscillatore_anharmonico,
        param_names=["A", "w", "offset"],
        p0=p0_sballato,
        limits={"A": (0.1, 10.0), "w": (2.0, 30.0), "offset": (-5.0, 5.0)},
        run_minos=True,
        verbose=True,
    )

    res_stress = pip_stress.fit(x_st, y_st, err_x=sx_st, err_y=sy_st)

    # Confronto visivo tra la curva di partenza (p0) e la curva convergente
    fig5, ax = plt.subplots(figsize=(9, 5))
    x_dense_st = np.linspace(0.0, 3.0, 400)
    ax.errorbar(x_st, y_st, xerr=sx_st, yerr=sy_st, fmt='.', color='black', alpha=0.4, label='Dati')
    ax.plot(x_dense_st, oscillatore_anharmonico(x_dense_st, *p0_sballato), 'r:', lw=1.5, label='Guess iniziale errato $p_0$')
    ax.plot(x_dense_st, oscillatore_anharmonico(x_dense_st, *res_stress.valori()), 'b-', lw=2.0, label='Fit finale dopo ricerca globale')
    ax.set_title("Recupero automatico del minimo da punto di partenza errato")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig5.tight_layout()
    fig5.savefig(f"{save_folder / 'esempio_5_recupero_globale.pdf'}", dpi=300, bbox_inches="tight")

    print("\nTutti gli esempi sono stati eseguiti e salvati correttamente.")
    plt.show()