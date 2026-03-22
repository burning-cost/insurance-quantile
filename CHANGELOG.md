# Changelog

## v0.3.3 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Make torch and lightgbm optional [eqrn] extras (v0.3.3)
- Add blog post link and community CTA to README
- fix: add MeanModelWrapper, tighten TVaR trapz test, bump to v0.3.2
- docs: add TwoPartQuantilePremium README section; restore full remote source
- Add TwoPartQuantilePremium: frequency × severity quantile premium (v0.3.0)
- Fix: lazy-import EQRN classes so torch is not required at package import time
- Fix: add TypeError with helpful message for non-Polars input to predict_premium()
- feat: add TwoPartQuantilePremium for frequency-severity QPP decomposition
- Add MIT license
- test: widen TVaR analytical tolerance to ±1.5 for limited-iteration fixture
- fix: TVaR boundary, multi-column validation, test label
- fix: QA audit P0/P1 fixes — TVaR integration, CatBoost Polars compat, coherence docs
- refresh benchmark numbers post-P0 fixes
- Fix P0/P1 bugs: ogpd negative xi clamp, array xi log density, OEP misnaming, predict_tvar dead code (v0.2.3)
- Add benchmark: QuantileGBM vs parametric lognormal quantiles
- fix: add numpy<2.0 compat shim for np.trapezoid in eqrn/gpd.py
- docs: add Databricks notebook link
- fix: handle numpy array input in fit() - columns attribute and _to_numpy
- fix: accept numpy arrays in _to_numpy and _series_to_numpy
- Add Related Libraries section to README
- fix: README quick-start blocks crash on numpy input — wrap in pl.DataFrame/pl.Series
- fix: define all variables in quick-start; correct polars version

