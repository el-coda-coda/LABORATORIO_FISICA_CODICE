"""
regressione automatica con ODR / curve_fit + iminuit + MINOS.

Flusso:
  1. Se err_x è nullo → curve_fit  (solo err_y)
     Altrimenti       → scipy ODR  (err_x ed err_y)
  2. I parametri stimati alimentano iminuit come punto di partenza.
  3. Il costo iminuit replica il costo ODR tramite minimizzazione ortogonale punto a punto (oppure tramite varianza effettiva se odr_lin_Newton=False).
  4. Dopo migrad() + hesse(), vengono calcolati gli errori MINOS per ogni
     parametro libero e confrontati con gli errori HESSE (simmetria percentuale).
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np
from iminuit import Minuit
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import least_squares

# ──────────────────────────────────────────────────────────────────────────────
# Struttura dati di output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParametroFit:
    # Stampa per il risultato completo per un singolo parametro
    nome:        str
    valore:      float
    err_hesse:   float           # errore simmetrico da HESSE
    err_minos_up:   float = np.nan  # errore superiore da MINOS  (+)
    err_minos_down: float = np.nan  # errore inferiore da MINOS  (-)
    minos_ok:    bool  = False
    asimmetria_assoluta: float = np.nan   # |up| - |down|
    asimmetria_pct:      float = np.nan   # (|up| - |down|) / media * 100

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
    
    # Tutti i risultati restituiti
    
    parametri:   list[ParametroFit]
    chi2:        float
    chi2_rid:    float
    dof:         int
    p_value:     float       
    metodo_init: str          # 'curve_fit' o 'ODR'
    valid:       bool
    covariance:  np.ndarray
    residui:     np.ndarray
    minuit_obj:  Minuit        # accesso all'oggetto Minuit

    # accesso per nome
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
# Funzione di costo ODR per iminuit
# ──────────────────────────────────────────────────────────────────────────────

# Questa classe implementa la funzione di costo ODR utilizzando l'approssimazione di linearizzazione per l'errore totale
class _CostoODR_linearizzato:
    """
    Costo ODR (chi-quadro effettivo) compatibile con iminuit.

    χ²_eff = Σ (y_i - f(x_i; θ))² / [σ_y² + (∂f/∂x · σ_x)²]

    Se σ_x ≡ 0 si riduce al χ² ordinario pesato su y.
    """
    errordef = Minuit.LEAST_SQUARES   # errordef = 1 → errori a 1σ

    def __init__(
        self,
        model:   Callable,
        x:       np.ndarray,
        y:       np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
    ):
        self._model   = model
        self._x       = x
        self._y       = y
        self._sx      = sigma_x
        self._sy      = sigma_y
        self._use_sx  = np.any(sigma_x > 0)

    def __call__(self, *params) -> float:
        y_pred = self._model(self._x, *params)
        if self._use_sx:
            
            # Calcolo numerico della derivata df/dx 
            x_scale = np.ptp(self._x) if np.ptp(self._x) > 0 else 1.0
            h = 1e-5 * (np.abs(self._x) + 0.01 * x_scale)
            dfdx   = (self._model(self._x + h, *params)
                      - self._model(self._x - h, *params)) / (2.0 * h)
            
            # Calcolo errore quadro per ogni punto con errore su x e y
            
            sigma2 = self._sy**2 + (dfdx * self._sx)**2
        else:
            
            # Calcolo errore quadro solo su y (nessun errore su x)
            
            sigma2 = self._sy**2
        return float(np.sum((self._y - y_pred)**2 / sigma2))
    
# Questa classe implementa la funzione di costo ODR completa, risolvendo iterativamente per gli scostamenti ortogonali delta_i, che convergono velocemente
class _CostoODR:
    """
    Costo ODR compatibile con iminuit.

    Minimizza numericamente gli scostamenti ortogonali delta_i per ogni punto:
      chi2 = sum( (delta_i / sx_i)^2 + ((y_i - f(x_i + delta_i)) / sy_i)^2 )

    Se sx == 0 si riduce istantaneamente al chi2 ordinario su y.
    """
    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        model: Callable,
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
        max_iter_delta: int = 4,
    ):
        self._model = model
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._sx = np.asarray(sigma_x, dtype=float)
        self._sy = np.asarray(sigma_y, dtype=float)
        self._use_sx = np.any(self._sx > 0)
        self._max_iter = max_iter_delta

        # Pre-calcolo pesi e passo numerico
        if self._use_sx:
            self._var_ratio = np.where(self._sy > 0, (self._sx / self._sy)**2, 0.0)
            x_scale = np.ptp(self._x) if np.ptp(self._x) > 0 else 1.0
            self._h = 1e-5 * (np.abs(self._x) + 0.01 * x_scale)

    # Metodo di Newton iterativo per minimizzare la distanza ortogonale punto per punto
    def _trova_delta(self, params: tuple) -> tuple[np.ndarray, np.ndarray]:
        """
        Risolve la proiezione ortogonale usando il passo di Newton-Raphson 1D:
          delta_{k+1} = delta_k - S'(delta_k) / S''(delta_k)
        garantendo convergenza anche con forti pendenze o errori elevati.
        """
        delta = np.zeros_like(self._x)
        inv_sx2 = 1.0 / (self._sx**2)
        inv_sy2 = 1.0 / (self._sy**2)

        for _ in range(self._max_iter):
            x_eval = self._x + delta
            y_pred = self._model(x_eval, *params)
            
            # Derivata df/dx nel punto proiettato
            dfdx = (self._model(x_eval + self._h, *params) - 
                    self._model(x_eval - self._h, *params)) / (2.0 * self._h)
            
            residuo_y = self._y - y_pred
            
            # Derivata prima S'(delta) e seconda S''(delta)
            grad = delta * inv_sx2 - residuo_y * dfdx * inv_sy2
            hess = inv_sx2 + (dfdx**2) * inv_sy2
            
            # Passo di Newton puro con bound di sicurezza sullo step
            step = grad / np.maximum(hess, 1e-12)
            delta = delta - step

        # Punti ortogonali finali
        y_proj = self._model(self._x + delta, *params)
        return delta, y_proj

    def __call__(self, *params) -> float:
        if not self._use_sx:
            y_pred = self._model(self._x, *params)
            return float(np.sum(((self._y - y_pred) / self._sy)**2))

        # Calcola gli scostamenti delta_i ottimali per questi parametri params
        delta, y_proj = self._trova_delta(params)

        # Somma quadratica delle deviazioni su x e su y
        term_x = (delta / self._sx)**2
        term_y = ((self._y - y_proj) / self._sy)**2

        # Gestione punti dove sx == 0
        term_x = np.nan_to_num(term_x, nan=0.0, posinf=0.0, neginf=0.0)

        return float(np.sum(term_x + term_y))


# ──────────────────────────────────────────────────────────────────────────────
# Classe di fit principale
# ──────────────────────────────────────────────────────────────────────────────

class FitPipeline:
    """
    Regressione automatica ODR / curve_fit  poi  iminuit (MINOS).

    Parametri
    ----------
    model : callable
        Firma attesa: ``model(x, p0, p1, ..., pN)``  (stile curve_fit / scipy).
    param_names : sequenza di stringhe
        Nomi dei parametri, nell'ordine di model.
    p0 : sequenza di float
        Valori iniziali (usati come fallback se ODR/curve_fit falliscono).
    limits : dict, opzionale
        ``{nome: (min, max)}`` passato a iminuit.
    fixed : dict, opzionale
        ``{nome: True/False}`` per fissare parametri.
    run_minos : bool
        Se True (default) calcola gli errori MINOS dopo migrad+hesse.
    verbose : bool
        Stampa il summary a fine fit.

    Uso
    ---
    >>> pip = FitPipeline(model, ['A', 'omega', 'phi'], p0=[1., 1e6, 0.])
    >>> res = pip.fit(x, y, err_x=np.zeros_like(x), err_y=sigma_y)
    >>> print(res)
    """

    def __init__(
        self,
        model:        Callable,
        param_names:  Sequence[str],
        p0:           Sequence[float],
        limits:       Optional[dict] = None,
        fixed:        Optional[dict] = None,
        run_minos:    bool = True,
        verbose:      bool = True,
    ):
        self._model        = model
        self._param_names  = list(param_names)
        self._p0           = np.asarray(p0, dtype=float)
        self._limits       = limits or {}
        self._fixed        = fixed  or {}
        self._run_minos    = run_minos
        self._verbose      = verbose

        # Check che il numero di parametri e nomi associati coincida

        if len(self._param_names) != len(self._p0):
            raise ValueError(
                f"param_names ({len(self._param_names)}) e p0 ({len(self._p0)}) "
                "devono avere la stessa lunghezza."
            )

    # ──────────────────────────────────────────────────────────────────────────────
    #  Metodo pubblico principale                                          
    # ──────────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        x:     np.ndarray,
        y:     np.ndarray,
        err_x: Optional[np.ndarray] = None,
        err_y: Optional[np.ndarray] = None,
        odr_lin_Newton: bool = True # True -> ODR Newton, False -> ODR linearizzato 
    ) -> RisultatiFit:
        """
        Esegue la pipeline completa e restituisce un oggetto RisultatiFit.

        Parametri
        ----------
        x, y     : array dei dati
        err_x    : incertezze su x  (None oppure array di zeri -> curve_fit)
        err_y    : incertezze su y  (None -> pesi uniformi)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        err_x = (np.zeros_like(x) if err_x is None
                 else np.asarray(err_x, dtype=float))
        err_y = (np.ones_like(y)  if err_y is None
                 else np.asarray(err_y, dtype=float))

        # Utilizza il fit ODR se almeno un errore su x è > 0
        
        use_odr = np.any(err_x > 0)

        # Stima iniziale utilizzando un metodo leggero
        p_init, metodo = self._stima_iniziale(x, y, err_x, err_y, use_odr)

        # Assegnazione della funzione di costo
        if odr_lin_Newton:
            cost = _CostoODR(self._model, x, y, err_x, err_y)
        else:
            cost = _CostoODR_linearizzato(self._model, x, y, err_x, err_y)

        # Fit utilizzando iminuit utilizzando la funzione di costoe utilizzando la stima iniziale (primo try con metodo HESSE)
        m = self._run_iminuit(cost, p_init)

        # Risultati del metodo MINOS
        results = self._build_results(m, x, y, err_x, err_y, metodo)

        # Stampa del summary default se impostata la modalità verbose = True

        if self._verbose:
            print(results.summary())

        return results

    # ──────────────────────────────────────────────────────────────────────────────
    #  Stima iniziale (curve_fit o ODR)
    # ──────────────────────────────────────────────────────────────────────────────

    def _stima_iniziale(
        self,
        x: np.ndarray, y: np.ndarray,
        err_x: np.ndarray, err_y: np.ndarray,
        use_odr: bool,
    ) -> tuple[np.ndarray, str]:

        # Se use_odr è True, utilizza ODR per la stima iniziale, altrimenti curve_fit
        
        if use_odr:
            return self._fit_odr(x, y, err_x, err_y)
        else:
            return self._fit_curve(x, y, err_y)

    # Fit con curve_fit (solo err_y)
    
    def _fit_curve(
        self,
        x: np.ndarray, y: np.ndarray, err_y: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        try:
            popt, _ = curve_fit(
                self._model, x, y,
                p0=self._p0,
                sigma=err_y,
                absolute_sigma=True,
                maxfev=50_000,
            )
            return popt, "curve_fit"
        except Exception as exc:
            warnings.warn(f"curve_fit fallito ({exc}); uso p0 come punto di partenza.")
            return self._p0.copy(), "curve_fit (p0 fallback)"

    # Fit con ODR (err_x ed err_y)

    def _fit_odr(
        self,
        x: np.ndarray, y: np.ndarray,
        err_x: np.ndarray, err_y: np.ndarray,
    ) -> tuple[np.ndarray, str]:

        # scipy ODR vuole model(beta, x) — wrapper
        def _odr_model(beta, t):
            return self._model(t, *beta)

        try:
            data = RealData(x, y, sx=err_x, sy=err_y)
            odr  = ODR(data, Model(_odr_model), beta0=self._p0)
            out  = odr.run()
            return out.beta, "ODR"
        except Exception as exc:
            warnings.warn(f"ODR fallito ({exc}); uso p0 come punto di partenza.")
            return self._p0.copy(), "ODR (p0 fallback)"

    # ──────────────────────────────────────────────────────────────────────────────
    #  Fit iminuit HESSE
    # ──────────────────────────────────────────────────────────────────────────────

    def _run_iminuit(self, cost: _CostoODR, p_init: np.ndarray) -> Minuit:
        m = Minuit(cost, *p_init, name=self._param_names)

        for nome, lim in self._limits.items():
            m.limits[nome] = lim
        for nome, fix in self._fixed.items():
            m.fixed[nome] = fix

        m.migrad()
        # Calcolo degli errori HESSE (simmetrici, ipotesi di simmetria del chi-quadro)
        m.hesse()

        if not m.valid:
            warnings.warn("iminuit: il fit non è valido (m.valid = False).")

        return m

    # ──────────────────────────────────────────────────────────────────────────────
    #  Fit iminuit MINOS e scrittura dei risultati
    # ──────────────────────────────────────────────────────────────────────────────

    def _build_results(
        self,
        m:       Minuit,
        x:       np.ndarray,
        y:       np.ndarray,
        err_x:   np.ndarray,
        err_y:   np.ndarray,
        metodo:  str,
    ) -> RisultatiFit:

        parametri: list[ParametroFit] = []

        for nome in self._param_names:
            valore    = m.values[nome]
            err_hesse = m.errors[nome]

            par = ParametroFit(
                nome=nome,
                valore=valore,
                err_hesse=err_hesse,
            )

            # MINOS solo per parametri liberi
            if self._run_minos and not m.fixed[nome]:
                try:
                    m.minos(nome)
                    me   = m.merrors[nome]
                    up   =  me.upper   # positivo
                    down = -me.lower   # reso positivo

                    media    = (up + down) / 2.0
                    asim_abs = up - down
                    asim_pct = (asim_abs / media * 100.0
                                if media > 0 else float('nan'))

                    par.err_minos_up          = up
                    par.err_minos_down        = down
                    par.minos_ok              = True
                    par.asimmetria_assoluta   = asim_abs
                    par.asimmetria_pct        = asim_pct

                except Exception as exc:
                    warnings.warn(f"MINOS fallito per '{nome}': {exc}")

            parametri.append(par)

        # Residui, chi2 e p-value
        y_pred   = self._model(x, *m.values)
        residui  = y - y_pred
        dof      = len(x) - m.nfit
        chi2_val = float(m.fval)
        chi2_rid = chi2_val / dof if dof > 0 else float('nan')
        p_val = chi2_dist.sf(chi2_val, dof) if dof > 0 else float('nan')
        
        # Check matrice di covarianza
        # Estrazione sicura della matrice di covarianza
        if m.covariance is not None:
            cov_matrix = np.array(m.covariance)
        else:
            warnings.warn(
                "[ERRORE COVARIANZA] Impossibile calcolare la matrice di covarianza: "
                "il fit non è convergente o l'Hessiana non è definita positiva.",
                RuntimeWarning,
                stacklevel=2
            )
            # Fallback a una matrice NaN (o di zeri) delle dimensioni attese
            cov_matrix = np.full((m.nfit, m.nfit), np.nan)

        return RisultatiFit(
            parametri   = parametri,
            chi2        = chi2_val,
            chi2_rid    = chi2_rid,
            dof         = dof,
            p_value     = p_val,
            metodo_init = metodo,
            valid       = m.valid,
            covariance  = cov_matrix,
            residui     = residui,
            minuit_obj  = m,
        )
    
# ──────────────────────────────────────────────────────────────────────────────
# Funzione di costo GLOBALE (Puro iminuit)
# ──────────────────────────────────────────────────────────────────────────────

class _CostoGlobale:
    """
    Cost function globale che somma i chi-quadri di N dataset.
    Compatibile al 100% con iminuit.
    """
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, models, xs, ys, err_xs, err_ys, odr_lin_Newton: bool = True):
        self.costi = []
        cost_cls = _CostoODR if odr_lin_Newton else _CostoODR_linearizzato
        for mod, x, y, sx, sy in zip(models, xs, ys, err_xs, err_ys):
            self.costi.append(cost_cls(mod, x, y, sx, sy))

    def __call__(self, *params) -> float:
        # iminuit varierà *params. Si calcola il chi2 di ogni modello con questi parametri e si restituisce la somma.
        return sum(costo(*params) for costo in self.costi)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Globale (SciPy pre-fit + iminuit)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalFitPipeline:
    """
    Regressione simultanea su N dataset.
    Flusso:
      1. least_squares (scipy) cerca una stima iniziale robusta.
      2. I parametri trovati vengono passati a iminuit.
      3. iminuit minimizza la funzione di costo globale (ODR-compatibile)
         e calcola gli errori HESSE/MINOS.
    """
    def __init__(
        self,
        models:       Sequence[Callable],
        param_names:  Sequence[str],
        p0:           Sequence[float],
        limits:       Optional[dict] = None,
        fixed:        Optional[dict] = None,
        run_minos:    bool = True,
        verbose:      bool = True,
    ):
        self._models       = models
        self._param_names  = list(param_names)
        self._p0           = np.asarray(p0, dtype=float)
        self._limits       = limits or {}
        self._fixed        = fixed  or {}
        self._run_minos    = run_minos
        self._verbose      = verbose

    def _stima_iniziale_globale(self, xs, ys, err_ys) -> tuple[np.ndarray, str]:
        """Usa scipy.optimize.least_squares per trovare un buon punto di partenza."""
        
        # Funzione obiettivo per least_squares: residui concatenati e pesati
        def residui_obj(p):
            res = []
            for mod, x, y, sy in zip(self._models, xs, ys, err_ys):
                y_pred = mod(x, *p)
                res.append((y - y_pred) / sy)
            return np.concatenate(res)

        # Traduzione dei limiti per least_squares (separati in array min e max)
        bounds_lower = [-np.inf] * len(self._p0)
        bounds_upper = [ np.inf] * len(self._p0)
        
        for i, name in enumerate(self._param_names):
                
            if name in self._limits and self._limits[name] is not None:
                vmin, vmax = self._limits[name]
                bounds_lower[i] = vmin if vmin is not None else -np.inf
                bounds_upper[i] = vmax if vmax is not None else np.inf

        try:
            out = least_squares(
                residui_obj, 
                x0=self._p0, 
                bounds=(bounds_lower, bounds_upper), 
                method='trf'
            )
            return out.x, "least_squares_globale"
        except Exception as e:
            import warnings
            warnings.warn(f"Stima iniziale con least_squares fallita: {e}. Uso p0 fornito.")
            return self._p0.copy(), "p0_fallback"

    def fit(
        self,
        xs:     Sequence[np.ndarray],
        ys:     Sequence[np.ndarray],
        err_xs: Optional[Sequence[np.ndarray]] = None,
        err_ys: Optional[Sequence[np.ndarray]] = None,
        odr_lin_Newton: bool = True  # True -> ODR Newton, False -> ODR linearizzato
    ) -> RisultatiFit:
        
        # Preparazione dati: conversione in array numpy e gestione default errori
        
        xs = [np.asarray(x, dtype=float) for x in xs]
        ys = [np.asarray(y, dtype=float) for y in ys]
        err_xs = [np.zeros_like(x) for x in xs] if err_xs is None else [np.asarray(sx, dtype=float) for sx in err_xs]
        err_ys = [np.ones_like(y) for y in ys]  if err_ys is None else [np.asarray(sy, dtype=float) for sy in err_ys]

        # Stima iniziale globale con least_squares e scipy
        p_init, metodo_init = self._stima_iniziale_globale(xs, ys, err_ys)

        # Stima con iminuit usando la funzione di costo globale
        cost = _CostoGlobale(self._models, xs, ys, err_xs, err_ys, odr_lin_Newton=odr_lin_Newton)
        
        # Minuit inizializzato con p_init (i valori trovati da SciPy)
        m = Minuit(cost, *p_init, name=self._param_names)
        
        for nome, lim in self._limits.items():
            m.limits[nome] = lim
        for nome, fix in self._fixed.items():
            m.fixed[nome] = fix

        # Esecuzione del fit globale con iminuit
        m.migrad()
        # Calcolo degli errori HESSE (simmetrici, ipotesi di simmetria del chi2)
        m.hesse()

        if not m.valid:
            import warnings
            warnings.warn("iminuit: il fit globale non è convergente (m.valid = False).")
            
        # Estrazione sicura della matrice di covarianza
        if m.covariance is not None:
            cov_matrix = np.array(m.covariance)
        else:
            warnings.warn(
                "[ERRORE COVARIANZA] Impossibile calcolare la matrice di covarianza: "
                "il fit non è convergente o l'Hessiana non è definita positiva.",
                RuntimeWarning,
                stacklevel=2
            )
            # Fallback a una matrice NaN (o di zeri) delle dimensioni attese
            cov_matrix = np.full((m.nfit, m.nfit), np.nan)

        # Costruzione dei risultati finali
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
                    print(f"  [Avviso] MINOS fallito per '{nome}': {exc}")
            parametri.append(par)

        # Calcolo residui totali concatenati
        residui_list = []
        n_punti_tot = 0
        for mod, x, y in zip(self._models, xs, ys):
            y_pred = mod(x, *m.values)
            residui_list.append(y - y_pred)
            n_punti_tot += len(x)

        # Calcolo chi2, dof e p-value globali
        dof = n_punti_tot - m.nfit
        chi2_val = float(m.fval)
        chi2_rid = chi2_val / dof if dof > 0 else float('nan')
        p_val = chi2_dist.sf(chi2_val, dof) if dof > 0 else float('nan')

        res = RisultatiFit(
            parametri   = parametri,
            chi2        = chi2_val,
            chi2_rid    = chi2_rid,
            dof         = dof,
            p_value     = p_val,
            metodo_init = f"iminuit (via {metodo_init})",
            valid       = m.valid,
            covariance  = cov_matrix,
            residui     = np.concatenate(residui_list),
            minuit_obj  = m,
        )

        if self._verbose:
            print(res.summary())

        return res

# ──────────────────────────────────────────────────────────────────────────────
# Esempio d'uso della classe FitPipeline con dati sintetici
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 1: Oscillazione smorzata standard
    # ──────────────────────────────────────────────────────────────────────────────
    
    def smorzata(x, A, delta, omega, phi):
        return A * np.exp(-delta * x) * np.cos(omega * x + phi)

    # Dati sintetici con rumore
    rng  = np.random.default_rng(42)
    t    = np.linspace(0, 4e-5, 400)
    A_t, d_t, w_t, p_t = 1.5, 8e4, 5e5, 0.3
    y_t  = smorzata(t, A_t, d_t, w_t, p_t)
    s_y  = 0.05 * np.max(np.abs(y_t))
    s_x  = 2e-8
    y_n  = y_t + rng.normal(0, s_y, len(t))
    x_n  = t   + rng.normal(0, s_x, len(t))

    # Pipeline con errori su x (ODR)
    pip = FitPipeline(
        model        = smorzata,
        param_names  = ["A", "delta", "omega", "phi"],
        p0           = [1.0, 1e5, 4e5, 0.0],
        limits       = {"A": (0, None), "delta": (0, None)},
        run_minos    = True,
        verbose      = True,
    )

    res = pip.fit(x_n, y_n, err_x=np.full_like(t, s_x), err_y=np.full_like(t, s_y))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    ax1.errorbar(t, y_n, yerr=s_y, fmt='.', alpha=0.4, label='dati')
    ax1.plot(t, smorzata(t, *res.valori()), 'r-', lw=1.5, label='fit')
    ax1.set_ylabel('y'); ax1.legend()
    ax2.errorbar(t, res.residui, yerr=s_y, fmt='.', alpha=0.4)
    ax2.axhline(0, color='r', ls='--'); ax2.set_ylabel('residui'); ax2.set_xlabel('t [s]')
    plt.tight_layout()
    
    # Definisci cartella e nome file
    folder = Path("examples")
    filename = "fit_pipeline_esempio.pdf"

    # Se la cartella esiste ed è una directory, usa il percorso completo, altrimenti salva nella directory corrente
    if folder.is_dir():
        save_path = folder / filename
    else:
        save_path = filename

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("\nGrafico salvato.")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # ESEMPIO 2: Confronto ODR Newton vs ODR Linearizzato (Effective Variance)
    # ──────────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 75)
    print("  ESEMPIO 2: Confronto Newton vs Linearizzato (Evidenza Bias)")
    print("=" * 75)

    def sinusoide(x, A, omega, phi):
        return A * np.sin(omega * x + phi)

    # Parametri veri
    A_true, w_true, phi_true = 2.0, 14.0, 0.5
    N_pts = 120
    x_exact = np.linspace(0, 1.2, N_pts)
    y_exact = sinusoide(x_exact, A_true, w_true, phi_true)

    # Incertezze: introduciamo un errore significativo su x per forzare la discrepanza
    sigma_x_val = 0.035
    sigma_y_val = 0.15
    sx_arr = np.full(N_pts, sigma_x_val)
    sy_arr = np.full(N_pts, sigma_y_val)

    x_data = x_exact + rng.normal(0, sigma_x_val, N_pts)
    y_data = y_exact + rng.normal(0, sigma_y_val, N_pts)

    pip2 = FitPipeline(
        model        = sinusoide,
        param_names  = ["A", "omega", "phi"],
        p0           = [1.8, 12.0, 0.0],
        run_minos    = False,
        verbose      = False,
    )

    # 1. Fit con metodo Newton (vero ODR ortogonale)
    res_newton = pip2.fit(x_data, y_data, err_x=sx_arr, err_y=sy_arr, odr_lin_Newton=True)

    # 2. Fit con metodo linearizzato (Effective Variance approssimata)
    res_lin = pip2.fit(x_data, y_data, err_x=sx_arr, err_y=sy_arr, odr_lin_Newton=False)

    # Tabella riassuntiva di confronto a terminale
    print(f"{'Parametro':<10} | {'Valore Vero':<12} | {'ODR Newton':<15} | {'Linearizzato':<15} | {'Diff (Lin - New)':<15}")
    print("-" * 75)
    for p_name in ["A", "omega", "phi"]:
        v_true = {"A": A_true, "omega": w_true, "phi": phi_true}[p_name]
        v_new = res_newton[p_name].valore
        v_lin = res_lin[p_name].valore
        diff = v_lin - v_new
        print(f"{p_name:<10} | {v_true:<12.4f} | {v_new:<15.4f} | {v_lin:<15.4f} | {diff:<+15.4f}")
    print("-" * 75)
    print(f"{'chi2 / dof':<10} | {'-':<12} | {res_newton.chi2_rid:<15.4f} | {res_lin.chi2_rid:<15.4f} | {res_lin.chi2_rid - res_newton.chi2_rid:<+15.4f}")
    print("=" * 75)

    # Grafico comparativo
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7),
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
    x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
    ax_top.errorbar(x_data, y_data, xerr=sx_arr, yerr=sy_arr, fmt='.', color='black', alpha=0.35, label='Dati con errori su x e y')
    ax_top.plot(x_plot, sinusoide(x_plot, *res_newton.valori()), 'b-', lw=2, label=f"ODR Newton (chi2/dof = {res_newton.chi2_rid:.2f})")
    ax_top.plot(x_plot, sinusoide(x_plot, *res_lin.valori()), 'r--', lw=1.5, label=f"Linearizzato (chi2/dof = {res_lin.chi2_rid:.2f})")
    ax_top.set_ylabel("y")
    ax_top.set_title("Confronto: ODR Newton vs ODR Linearizzato (bias su parametri di derivata)")
    ax_top.legend()

    # Differenza diretta tra i due fit nei residui
    ax_bot.plot(x_plot, sinusoide(x_plot, *res_lin.valori()) - sinusoide(x_plot, *res_newton.valori()), 'm-', lw=1.5, label='Diff curve (Lin - New)')
    ax_bot.axhline(0, color='gray', ls=':')
    ax_bot.set_ylabel("Delta Fit")
    ax_bot.set_xlabel("x")
    ax_bot.legend()
    fig2.tight_layout()
    
    # Definisci cartella e nome file
    folder = Path("examples")
    filename = "fit_odr_difference.pdf"

    # Se la cartella esiste ed è una directory, usa il percorso completo, altrimenti salva nella directory corrente
    if folder.is_dir():
        save_path = folder / filename
    else:
        save_path = filename
        
    fig2.savefig(save_path, dpi=300, bbox_inches="tight")

    print("\nGrafici degli esempi salvati con successo.")
    plt.show()