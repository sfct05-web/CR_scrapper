import os
import re
from io import StringIO
from typing import Dict, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    raise RuntimeError("Tu Python no soporta zoneinfo. Usa Python 3.9+.")


URL = "https://apps.grupoice.com/CenceWeb/CenceBoletinIntraDiario.jsf?init=true"
OUT_DIR = "data"
TZ = ZoneInfo("America/Guatemala")


TECH_KEYWORDS = {
    "hidrico": "Hídrico",
    "hídrico": "Hídrico",
    "solar": "Solar",
    "biomasa": "Biomasa",
    "termico": "Térmico",
    "térmico": "Térmico",
    "geotermico": "Geotérmico",
    "geotérmico": "Geotérmico",
    "eolico": "Eólico",
    "eólico": "Eólico",
    "intercambio": "Intercambio",
}


def _slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return (s[:max_len] or "tabla").strip("_") or "tabla"


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("\u00a0", " ")
    # coma decimal
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    # separador de miles
    if "," in s and "." in s:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extraer_tablas_a_dfs(url: str, timeout_ms: int = 60000) -> Dict[str, pd.DataFrame]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Bloquear recursos pesados para eficiencia
        def route_handler(route):
            rtype = route.request.resource_type
            if rtype in ("image", "media", "font"):
                route.abort()
            else:
                route.continue_()

        page.route("**/*", route_handler)

        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_selector("table", timeout=timeout_ms)
        page.wait_for_timeout(500)

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError("No se encontraron <table> en el DOM renderizado.")

    out: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tables, start=1):
        table_html = str(t)

        caption = t.find("caption")
        caption_txt = caption.get_text(" ", strip=True) if caption else ""
        tid = t.get("id", "") or t.get("name", "") or ""
        aria = t.get("aria-label", "") or t.get("summary", "") or ""

        nearby_title = ""
        prev = t.find_previous(["h1", "h2", "h3", "h4", "label", "span", "div"])
        if prev:
            txt = prev.get_text(" ", strip=True)
            if txt and len(txt) <= 120:
                nearby_title = txt

        base_name = caption_txt or aria or nearby_title or tid or f"tabla_{i}"
        name = f"{_slug(base_name)}_{i:02d}"

        try:
            dfs = pd.read_html(StringIO(table_html))
            if dfs:
                out[name] = dfs[0]
        except ValueError:
            continue

    if not out:
        raise RuntimeError("Se detectaron <table> pero ninguna se pudo convertir a DataFrame.")

    return out


def _infer_tecnologia(table_name: str, df: pd.DataFrame) -> Optional[str]:
    name = (table_name or "").lower()
    for k, v in TECH_KEYWORDS.items():
        if k in name:
            return v

    cols = " ".join([str(c) for c in df.columns]).lower()
    for k, v in TECH_KEYWORDS.items():
        if k in cols:
            return v

    if len(df.columns) > 0:
        c0 = str(df.columns[0]).lower()
        for k, v in TECH_KEYWORDS.items():
            if k in c0:
                return v
    return None


def _pick_name_value_columns(df: pd.DataFrame):
    df2 = df.copy()

    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df2.columns
        ]

    df2.columns = [str(c).strip() for c in df2.columns]

    value_candidates = []
    name_candidates = []

    for c in df2.columns:
        lc = str(c).lower()
        if "mwh" in lc:
            value_candidates.append(c)
        if all(x not in lc for x in ["mwh", "programado", "real", "msnm"]):
            name_candidates.append(c)

    value_col = value_candidates[0] if value_candidates else None
    name_col = name_candidates[0] if name_candidates else None

    if value_col is None:
        best, best_score = None, -1
        for c in df2.columns:
            vals = df2[c].head(30).tolist()
            score = sum(_to_float(v) is not None for v in vals)
            if score > best_score:
                best_score = score
                best = c
        if best_score > 0:
            value_col = best

    if name_col == value_col:
        for c in df2.columns:
            if c != value_col:
                name_col = c
                break

    return df2, name_col, value_col


def tablas_a_df_limpio(tablas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    filas: List[pd.DataFrame] = []

    for tname, df in tablas.items():
        if df is None or df.empty:
            continue

        # ignorar embalses / msnm
        cols_join = " ".join([str(c).lower() for c in df.columns])
        if "embalse" in cols_join or "msnm" in cols_join:
            continue

        tecnologia = _infer_tecnologia(tname, df)
        if tecnologia is None:
            continue

        df2, name_col, value_col = _pick_name_value_columns(df)
        if not name_col or not value_col:
            continue

        tmp = df2[[name_col, value_col]].copy()
        tmp.columns = ["central_raw", "mwh_raw"]

        tmp["central"] = (
            tmp["central_raw"]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        tmp["mwh"] = tmp["mwh_raw"].apply(_to_float)

        tmp = tmp[tmp["central"].ne("")]
        tmp = tmp[tmp["mwh"].notna()]
        tmp = tmp.drop_duplicates(subset=["central", "mwh"])

        tmp["tecnologia"] = tecnologia
        filas.append(tmp[["tecnologia", "central", "mwh"]])

    if not filas:
        return pd.DataFrame(columns=["tecnologia", "central", "mwh"])

    out = pd.concat(filas, ignore_index=True)
    out = out.sort_values(["tecnologia", "central"], ignore_index=True)
    return out


def monthly_csv_path(now_local: datetime) -> str:
    # archivo mensual basado en Guatemala
    ym = now_local.strftime("%Y-%m")
    return os.path.join(OUT_DIR, f"cence_generacion_{ym}.csv")


def append_historico_mensual(df_nuevo: pd.DataFrame, out_csv: str, now_local: datetime) -> pd.DataFrame:
    """
    Agrega:
    - fecha_iso: YYYY-MM-DD (Guatemala)
    - hora_consulta: HH:MM:SS (Guatemala)
    - timestamp_local_iso: ISO completo con offset (Guatemala)
    y hace append al CSV mensual.
    """
    df_nuevo = df_nuevo.copy()

    df_nuevo["fecha_iso"] = now_local.date().isoformat()
    df_nuevo["hora_consulta"] = now_local.strftime("%H:%M:%S")
    df_nuevo["timestamp_local_iso"] = now_local.isoformat(timespec="seconds")
    df_nuevo["source_url"] = URL

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if os.path.exists(out_csv):
        hist = pd.read_csv(out_csv)
        combinado = pd.concat([hist, df_nuevo], ignore_index=True)

        # dedup por corrida exacta
        combinado = combinado.drop_duplicates(
            subset=["timestamp_local_iso", "tecnologia", "central", "mwh"],
            keep="last",
        )
    else:
        combinado = df_nuevo

    combinado.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return combinado


def main():
    now_local = datetime.now(TZ).replace(microsecond=0)
    out_csv = monthly_csv_path(now_local)

    tablas = extraer_tablas_a_dfs(URL)
    df_gen = tablas_a_df_limpio(tablas)

    if df_gen.empty:
        raise RuntimeError("DF de generación salió vacío (no se detectaron tablas MWh parseables).")

    combinado = append_historico_mensual(df_gen, out_csv, now_local)

    print("OK")
    print("Hora Guatemala:", now_local.isoformat(timespec="seconds"))
    print("Archivo mensual:", out_csv)
    print("Nueva corrida filas:", len(df_gen))
    print("Histórico mensual filas:", len(combinado))
    print(combinado.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
