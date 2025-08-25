# -*- coding: utf-8 -*-
# dashboard_netflix.py

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

# =========================
# Configuração de página/UI
# =========================
st.set_page_config(page_title="Netflix Exec Dashboard — Reed Hastings", page_icon="🎬", layout="wide")

PRIMARY_COLOR = "#E50914"
CARD_BG = "#141414"
TEXT_MUTED = "#b3b3b3"

# ---- Estilos globais
st.markdown(f"""
<style>
html, body, [class*='css']  {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}}
.block-container {{ padding-top: 1.2rem; padding-bottom: 1.5rem; }}
h1, h2, h3, h4 {{ font-weight: 700; }}
.card {{
  background: {CARD_BG}; border: 1px solid #2a2a2a; border-radius: 18px;
  padding: 16px 18px; box-shadow: 0 8px 20px rgba(0,0,0,.20);
}}
.badge {{
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  background: {PRIMARY_COLOR}; color: #fff; font-weight: 600; font-size: 12px;
}}
.section-title {{ margin: 6px 0; color: {TEXT_MUTED}; font-size: 13px; letter-spacing: .2px; }}

/* remove ícones de âncora dos títulos */
h1 a, h2 a, h3 a {{ display: none !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{{ gap: 8px; margin-top: 6px; padding-bottom: 6px; }}
.stTabs [data-baseweb="tab"]{{
  background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:10px 14px;
  color:#cfcfcf;font-weight:600;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"]{{ background:#E50914;color:#fff;border-color:#E50914; }}

/* KPI cards */
.kpi-card{{
  background:#111; border:1px solid #2a2a2a; border-radius:16px;
  padding:14px 16px; box-shadow:0 6px 16px rgba(0,0,0,.20);
}}
.kpi-title{{font-size:13px;color:#c9c9c9;font-weight:600;letter-spacing:.2px}}
.kpi-value{{font-size:28px;font-weight:800;margin-top:4px}}
.kpi-foot{{font-size:12px;color:#9f9f9f;margin-top:2px}}
.subtle{{color:#9fa3a7;font-size:14px;margin-top:-6px}}
.chart-caption{{color:#a7a7a7;font-size:12px;margin:6px 0 16px 0}}

/* PERSONA mega-card */
.persona-card {{
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 18px;
    padding: 20px;
    display: flex;
    align-items: flex-start;
    gap: 30px;
    box-shadow: 0 8px 20px rgba(0,0,0,.25);
    margin-bottom: 20px;
}}
.persona-photo {{ flex: 0 0 240px; }}
.persona-photo img {{
    width: 240px; height: 240px; object-fit: cover;
    border-radius: 14px; border: 3px solid #E50914;
}}
.persona-info {{ flex: 1; color: #fff; }}
.persona-info h2 {{ color: #E50914; margin: 0; font-size: 30px; font-weight: 900; letter-spacing:.3px;}}
.persona-info h4 {{ margin: 6px 0 20px 0; color: #ccc; font-weight: 600; }}
.persona-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
}}
.persona-box {{
    background: #1f1f1f;
    border-radius: 14px;
    padding: 16px;
    color: #ddd;
    box-shadow: 0 6px 16px rgba(0,0,0,.18);
}}
.persona-box h4 {{ margin: 0 0 10px 0; color: #E7E7E7; font-weight: 800; }}
.persona-box ul {{ padding-left: 20px; margin: 0; }}
.persona-box li {{ margin-bottom: 6px; font-size: 14px; }}

@media(max-width:1100px){{ .persona-grid{{grid-template-columns:1fr 1fr;}} }}
@media(max-width:680px){{ .persona-card{{flex-direction:column;}} .persona-grid{{grid-template-columns:1fr;}} }}
</style>
""", unsafe_allow_html=True)

# Funções auxiliares de UI
def section(title: str, subtitle: str = "", emoji: str = ""):
    icon = f"{emoji} " if emoji else ""
    st.markdown(f"<h3>{icon}{title}</h3>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)

def kpi_card(icon: str, label: str, value: str, foot: str = ""):
    return f"""
    <div class='kpi-card'>
      <div class='kpi-title'>{icon} {label}</div>
      <div class='kpi-value'>{value}</div>
      {f"<div class='kpi-foot'>{foot}</div>" if foot else ""}
    </div>
    """

def center_plot(fig, caption: Optional[str] = None):
    left, mid, right = st.columns([0.07, 0.86, 0.07])
    with mid:
        st.plotly_chart(fig, use_container_width=True)
        if caption:
            st.markdown(f"<div class='chart-caption'>{caption}</div>", unsafe_allow_html=True)

# Plotly defaults
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Reds"

# ================= Constantes/dados
DATA_DIR = "data"
NETFLIX_FILENAME = "netflix_titles.csv"
NETFLIX_PATH = os.path.join(DATA_DIR, NETFLIX_FILENAME)
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
PERSONA_IMG = os.path.join(DATA_DIR, "reed_persona.png")

# ================= Helpers de dados
def _split_and_strip(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .apply(lambda x: [s.strip() for s in x.split(",") if s.strip()])
    )

def explode_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df.assign(**{col: []})
    temp = df.copy()
    temp[col] = _split_and_strip(temp[col])
    temp = temp.explode(col)
    temp[col] = temp[col].fillna("").astype(str).str.strip()
    return temp

@st.cache_data(show_spinner=False)
def load_netflix_data() -> pd.DataFrame:
    if not os.path.exists(NETFLIX_PATH):
        st.error(f"❌ Arquivo não encontrado: {NETFLIX_PATH}. Coloque '{NETFLIX_FILENAME}' na pasta '{DATA_DIR}/'.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(NETFLIX_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(NETFLIX_PATH, encoding="latin-1")
    except Exception as e:
        st.error(f"Erro ao ler {NETFLIX_PATH}: {e}")
        return pd.DataFrame()

    for col in ["country", "listed_in", "cast", "director", "title", "type", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    df["n_countries"] = df["country"].apply(lambda x: 0 if not x else len([c.strip() for c in str(x).split(",") if c.strip()])) if "country" in df.columns else 0
    df["n_genres"] = df["listed_in"].apply(lambda x: 0 if not x else len([g.strip() for g in str(x).split(",") if g.strip()])) if "listed_in" in df.columns else 0
    return df

@st.cache_data(show_spinner=False)
def load_ratings() -> Tuple[pd.DataFrame, Optional[str]]:
    if not os.path.exists(RATINGS_PATH):
        return pd.DataFrame(), None
    try:
        r = pd.read_csv(RATINGS_PATH)
    except UnicodeDecodeError:
        r = pd.read_csv(RATINGS_PATH, encoding="latin-1")
    r.columns = [c.strip() for c in r.columns]
    candidates = ["score", "imdb_score", "tmdb_score", "rating", "averageRating", "vote_average"]
    score_col = next((c for c in candidates if c in r.columns), None)
    if score_col is None:
        numeric_cols = r.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            s = r[c].dropna()
            if s.empty: continue
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            if (0 <= lo <= 10 and 0 <= hi <= 10) or (0 <= lo <= 100 and 0 <= hi <= 100):
                score_col = c; break
    return r, score_col

def ensure_year_range(df: pd.DataFrame) -> Tuple[int, int]:
    if "release_year" not in df.columns or df["release_year"].dropna().empty:
        return (1950, 2025)
    years = df["release_year"].dropna().astype(int)
    return (int(years.min()), int(years.max()))

# =========== Carrega dados
df_raw = load_netflix_data()
if df_raw.empty:
    st.stop()

ratings_df, ratings_col = load_ratings()
df = df_raw.copy()
if not ratings_df.empty and ratings_col:
    df["title_norm"] = df["title"].str.strip().str.lower()
    ratings_df["title_norm"] = ratings_df["title"].astype(str).str.strip().str.lower()
    df = df.merge(ratings_df[["title_norm", ratings_col]], on="title_norm", how="left")
    df.rename(columns={ratings_col: "score"}, inplace=True)
    if "score" in df.columns and df["score"].dropna().max() > 10:
        df["score"] = df["score"] / 10.0

# =========== Sidebar / Filtros
with st.sidebar:
    st.markdown("### 🎬 Netflix Exec Dashboard")
    st.caption("Persona: **Reed Hastings (CEO)** — foco em decisões de catálogo e expansão.")

    min_y, max_y = ensure_year_range(df)

    df_c = explode_column(df, "country") if "country" in df.columns else pd.DataFrame(columns=["country"])
    countries = sorted([c for c in df_c["country"].dropna().unique() if c])

    df_g = explode_column(df, "listed_in") if "listed_in" in df.columns else pd.DataFrame(columns=["listed_in"])
    genres = sorted([g for g in df_g["listed_in"].dropna().unique() if g])

    sel_countries = st.multiselect("🌍 País", options=countries, default=[])
    sel_genres = st.multiselect("🎭 Gênero", options=genres, default=[])
    year_range = st.slider("📅 Ano de lançamento", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1)

    score_available = "score" in df.columns and df["score"].notna().any()
    if score_available:
        lo = float(np.nanpercentile(df["score"], 1))
        hi = float(np.nanpercentile(df["score"], 99))
        score_range = st.slider("⭐ Faixa de avaliação (score)", 0.0, 10.0, (max(0.0, round(lo,1)), min(10.0, round(hi,1))), 0.1)
    else:
        score_range = None

def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    if "release_year" in d.columns and isinstance(year_range, (list, tuple)) and len(year_range) == 2:
        y0, y1 = year_range
        d = d[(d["release_year"] >= y0) & (d["release_year"] <= y1)]

    if sel_countries and "country" in d.columns:
        d = d[d["country"].apply(lambda x: any(c in [s.strip() for s in str(x).split(",") if s.strip()] for c in sel_countries))]

    if sel_genres and "listed_in" in d.columns:
        d = d[d["listed_in"].apply(lambda x: any(g in [s.strip() for s in str(x).split(",") if s.strip()] for g in sel_genres))]

    if ("score" in d.columns) and isinstance(score_range, (list, tuple)) and len(score_range) == 2:
        s0, s1 = score_range
        d = d[d["score"].between(s0, s1, inclusive="both")]

    return d

df_f = apply_filters(df)

# =========== Abas
tab1, tab2, tab4= st.tabs([
    "Ficha da Persona", "Empatia", "Dashboard"
])

# ===== Ficha da Persona — Dark elegante
# ===== Ficha da Persona — Fullscreen só imagem =====
with tab1:
    import base64

    def _img_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    img_path = os.path.join("data", "reed_persona.png")

    st.markdown("""
    <style>
      .persona-fullscreen {
        background-color: #121212;
        width: 100%;
        height: 100vh;  /* ocupa a tela inteira */
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .persona-img {
        max-width: 95%;
        max-height: 95%;
        border-radius: 10px;
        box-shadow: 0px 6px 25px rgba(0,0,0,0.6);
      }
    </style>
    """, unsafe_allow_html=True)

    if os.path.exists(img_path):
        img_b64 = _img_to_base64(img_path)
        st.markdown(f"""
        <div class="persona-fullscreen">
          <img src="data:image/png;base64,{img_b64}" class="persona-img" alt="Ficha da Persona">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Não encontrei `data/reed_persona.png`. Coloque a imagem na pasta /data.")



# ===== Etapa 2 — Empatia
# ===== Etapa 2 — Empatia (fullscreen só imagem) =====
with tab2:
    import base64

    def _img_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    img_path = os.path.join("data", "empatia.png")

    st.markdown("""
    <style>
      .empatia-fullscreen {
        background-color: #121212;
        width: 100%;
        height: 100vh;  /* ocupa a tela inteira */
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .empatia-img {
        max-width: 95%;
        max-height: 95%;
        border-radius: 10px;
        box-shadow: 0px 6px 25px rgba(0,0,0,0.6);
      }
    </style>
    """, unsafe_allow_html=True)

    if os.path.exists(img_path):
        img_b64 = _img_to_base64(img_path)
        st.markdown(f"""
        <div class="empatia-fullscreen">
          <img src="data:image/png;base64,{img_b64}" class="empatia-img" alt="Mapa de Empatia">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Não encontrei `data/empatia.png`. Coloque a imagem na pasta /data.")
# ===== Etapa 4 — Dashboard (UX executivo refinado)

# ================= Etapa 4 — Dashboard Executivo (reformulado) =================
# ================= Helpers de layout =================

# ===== Etapa 4 — Dashboard (reformulado, claro e centralizado)
with tab4:
    # --------- CSS leve para centralizar e melhorar contraste ---------
    st.markdown("""
        <style>
        :root { --bg:#ffffff; --text:#111418; --sub:#5f6b7a; --card:#f7f8fa; --border:#dfe3ea; --red:#E50914; }
        .nx-wrap { max-width:1100px; margin:0 auto; }
        .nx-subtle { color:var(--sub); font-size:.95rem; }
        .nx-block { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px 20px; margin:18px 0; }
        .nx-kpi h1{ font-size:2.1rem; margin:2px 0; color:var(--red); }
        .nx-kpi .label{ color:var(--sub); font-size:.92rem; }
        .nx-divider { border-top:1px solid var(--border); margin:18px 0; }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Cabeçalho e filtros ativos ----------
    filtros_text = []
    if sel_countries: filtros_text.append(f"País: {', '.join(sel_countries)}")
    if sel_genres: filtros_text.append(f"Gênero: {', '.join(sel_genres)}")
    filtros_text.append(f"Ano: {year_range[0]}–{year_range[1]}")
    if "score" in df.columns and isinstance(score_range, (list, tuple)) and df["score"].notna().any():
        filtros_text.append(f"Score: {score_range[0]:.1f}–{score_range[1]:.1f}")

    st.markdown("<div class='nx-wrap'>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Dashboard Netflix</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='nx-subtle'>Visão com filtros ativos — {' | '.join(filtros_text)}</div>", unsafe_allow_html=True)

    # ---------- Guard-clause ----------
    if df_f.empty:
        st.info("🧭 Nenhum título atende aos critérios atuais. **Amplie os filtros** (país/ano/gênero/score) para obter insights.")
        st.markdown("</div>", unsafe_allow_html=True)  # fecha .nx-wrap
        st.stop()

    # ---------- KPIs (centralizados) ----------
    total_titles = int(df_f.shape[0])
    n_countries = explode_column(df_f, "country")["country"].nunique() if "country" in df_f.columns else 0
    n_genres = explode_column(df_f, "listed_in")["listed_in"].nunique() if "listed_in" in df_f.columns else 0

    c1, c2, c3 = st.columns([1,1,1], gap="large")
    with c1:
        st.markdown("<div class='nx-block nx-kpi'>", unsafe_allow_html=True)
        st.markdown("**🎬 Títulos**")
        st.markdown(f"<h1>{total_titles:,}</h1>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Catálogo filtrado</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='nx-block nx-kpi'>", unsafe_allow_html=True)
        st.markdown("**🌍 Países**")
        st.markdown(f"<h1>{n_countries:,}</h1>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Cobertura geográfica</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='nx-block nx-kpi'>", unsafe_allow_html=True)
        st.markdown("**🎭 Gêneros**")
        st.markdown(f"<h1>{n_genres:,}</h1>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Variedade de conteúdo</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='nx-divider'></div>", unsafe_allow_html=True)

    # ========= 1) Onde estamos? =========
    st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
    st.markdown("### 🌍 Onde estamos?")
    top_ctry_caption = "Concentração de títulos por país."
    if "country" in df_f.columns:
        _c = explode_column(df_f, "country")
        if not _c.empty:
            tops = _c["country"].value_counts().head(3).index.tolist()
            if tops: top_ctry_caption = f"Concentração maior em **{', '.join(tops)}** — priorize presença/marketing."
    st.caption(top_ctry_caption)

    if "country" in df_f.columns:
        df_country = explode_column(df_f, "country")
        df_country = df_country[df_country["country"].astype(str).str.len() > 0]
        if not df_country.empty:
            cnt = df_country.groupby("country").size().reset_index(name="qtd").sort_values("qtd", ascending=False)
            fig_map = px.choropleth(cnt, locations="country", locationmode="country names",
                                    color="qtd", hover_name="country", color_continuous_scale="Reds")
            fig_map.update_coloraxes(colorbar_title="# de títulos")
            fig_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>Qtd: %{z}<extra></extra>")
            center_plot(fig_map)
    st.markdown("</div>", unsafe_allow_html=True)

    # Foco quando há único país
    if len(sel_countries) == 1:
        pais = sel_countries[0]
        st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
        st.markdown(f"#### 🎯 Foco em {pais}")
        colA, colB = st.columns([0.60, 0.40], gap="large")

        with colA:
            df_gen_pais = explode_column(df_f[df_f["country"].str.contains(pais, na=False)], "listed_in")
            vc = df_gen_pais["listed_in"].value_counts().head(15) if not df_gen_pais.empty else pd.Series(dtype=int)
            if not vc.empty:
                gen_cnt = pd.DataFrame({"Gênero": vc.index.astype(str), "Qtd": vc.values}).sort_values("Qtd", ascending=True)
                fig_pais_gen = px.bar(gen_cnt, x="Qtd", y="Gênero", orientation="h", labels={"Qtd":"Qtd","Gênero":"Gênero"})
                fig_pais_gen.update_layout(yaxis={"categoryorder":"total ascending"})
                fig_pais_gen.update_traces(text=gen_cnt["Qtd"], textposition="outside", cliponaxis=False)
                center_plot(fig_pais_gen, caption=f"Em **{pais}**, gêneros mais frequentes orientam promoções locais.")

        with colB:
            if "score" in df_f.columns and df_f["score"].notna().any():
                top_local = (df_f[df_f["country"].str.contains(pais, na=False)]
                             .dropna(subset=["score"]).sort_values("score", ascending=False).head(10))
                if not top_local.empty:
                    fig_top_local = px.bar(top_local.iloc[::-1], x="score", y="title", orientation="h",
                                           labels={"score":"Score","title":"Título"})
                    fig_top_local.update_traces(text=top_local.iloc[::-1]["score"].round(1),
                                                textposition="outside", cliponaxis=False)
                    leader = top_local.iloc[0]["title"]
                    center_plot(fig_top_local, caption=f"Top avaliados em **{pais}** — liderança: **{leader}**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ========= 2) O que o público consome? =========
    st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
    st.markdown("### 🎭 O que o público consome?")
    if "listed_in" in df_f.columns:
        df_gen = explode_column(df_f, "listed_in")
        df_gen = df_gen[df_gen["listed_in"].astype(str).str.len() > 0]
        if not df_gen.empty:
            vc = df_gen["listed_in"].value_counts().head(20)
            gen_cnt = pd.DataFrame({"Gênero": vc.index.astype(str), "Qtd": vc.values}).sort_values("Qtd", ascending=True)

            c1, c2 = st.columns([0.62, 0.38], gap="large")
            with c1:
                st.markdown("**Gêneros mais frequentes (Top 20)**")
                fig_barh = px.bar(gen_cnt, x="Qtd", y="Gênero", orientation="h", labels={"Qtd":"Qtd","Gênero":"Gênero"})
                fig_barh.update_layout(yaxis={"categoryorder":"total ascending"})
                fig_barh.update_traces(text=gen_cnt["Qtd"], textposition="outside", cliponaxis=False)
                leaders = ", ".join(gen_cnt.sort_values("Qtd", ascending=False)["Gênero"].head(3))
                center_plot(fig_barh, caption=f"**Líderes globais:** {leaders} — priorizar aquisição/destaque.")

            with c2:
                st.markdown("**Participação por gênero (Treemap)**")
                fig_tree = px.treemap(gen_cnt.sort_values("Qtd", ascending=False),
                                      path=["Gênero"], values="Qtd", labels={"Qtd":"Qtd"})
                center_plot(fig_tree, caption="Proporções evidenciam o peso de cada cluster.")

    # Heatmap País × Gênero
    st.markdown("<div class='nx-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
    st.markdown("**País × Gênero (Top 15 × Top 15)**")
    if {"country","listed_in"}.issubset(df_f.columns):
        tmp = df_f.copy()
        tmp["country"] = _split_and_strip(tmp["country"]); tmp = tmp.explode("country")
        tmp["listed_in"] = _split_and_strip(tmp["listed_in"]); tmp = tmp.explode("listed_in")
        tmp = tmp[(tmp["country"].astype(str)!="") & (tmp["listed_in"].astype(str)!="")]
        if not tmp.empty:
            top_c = tmp["country"].value_counts().head(15).index
            top_g = tmp["listed_in"].value_counts().head(15).index
            pv = tmp[tmp["country"].isin(top_c) & tmp["listed_in"].isin(top_g)] \
                    .pivot_table(index="country", columns="listed_in", aggfunc="size", fill_value=0).sort_index()
            if not pv.empty:
                fig_heat = px.imshow(pv, aspect="auto", labels=dict(color="# de títulos"))
                center_plot(fig_heat, caption="Quadrantes escuros = maior incidência; foque nesses cruzamentos.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= 3) Como evoluímos? =========
    st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
    st.markdown("### 📈 Como evoluímos?")
    if "release_year" in df_f.columns and df_f["release_year"].notna().any():
        yr = df_f.groupby("release_year").size().reset_index(name="Lançamentos").sort_values("release_year")
        if not yr.empty:
            fig_line = px.line(yr, x="release_year", y="Lançamentos", markers=True, labels={"release_year":"Ano"})
            fig_line.update_traces(hovertemplate="Ano %{x}<br>Qtd: %{y}<extra></extra>")
            trend = "crescimento recente" if yr["Lançamentos"].tail(3).is_monotonic_increasing else "volatilidade recente"
            center_plot(fig_line, caption=f"Tendência geral: **{trend}**. Ajuste aquisições ao calendário.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= 4) Quem se destaca? =========
    if "score" in df_f.columns and df_f["score"].notna().any():
        st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
        st.markdown("### ⭐ Quem se destaca?")
        top10 = df_f.dropna(subset=["score"]).sort_values("score", ascending=False).head(10)
        base_hist = df_f.dropna(subset=["score"])

        c1, c2 = st.columns([0.56, 0.44], gap="large")
        with c1:
            if not top10.empty:
                fig_top = px.bar(top10.iloc[::-1], x="score", y="title", orientation="h",
                                 labels={"score":"Score","title":"Título"})
                fig_top.update_traces(text=top10.iloc[::-1]["score"].round(1), textposition="outside", cliponaxis=False)
                cap = f"Média do Top 10: **{top10['score'].mean():.1f}** (faixa {top10['score'].min():.1f}–{top10['score'].max():.1f})."
                center_plot(fig_top, caption=cap)

        with c2:
            if not base_hist.empty:
                fig_hist = px.histogram(base_hist, x="score", nbins=25, labels={"score":"Score"})
                cap = f"Mediana do portfólio **{base_hist['score'].median():.1f}**; caudas indicam riscos/outliers."
                center_plot(fig_hist, caption=cap)

        if "release_year" in df_f.columns:
            base_scatter = df_f.dropna(subset=["score","release_year"])
            if not base_scatter.empty:
                fig_scatter = px.scatter(base_scatter, x="release_year", y="score", hover_name="title",
                                         labels={"release_year":"Ano","score":"Score"})
                center_plot(fig_scatter, caption="Score ao longo do tempo revela safras fortes e quedas.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
        st.markdown("### ℹ️ Avaliações indisponíveis")
        st.caption("Inclua um arquivo de avaliações (ex.: `data/ratings.csv`) para desbloquear análises de qualidade.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ========= 5) O que comunicamos? =========
    st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
    st.markdown("### 📝 O que comunicamos?")
    if "description" in df_f.columns and df_f["description"].notna().any():
        text = " ".join(df_f["description"].dropna().astype(str).tolist())
        if text.strip():
            extra_stop = set(["film","series","movie","netflix","season","year","story","one","two","new","set","based","life"])
            wc = WordCloud(width=1000, height=360, background_color="white", stopwords=STOPWORDS.union(extra_stop))
            img = wc.generate(text).to_image()
            st.image(img, caption="Termos dominantes nas descrições do catálogo.", use_column_width=True)
    else:
        st.caption("Sem descrições disponíveis no recorte atual.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= 6) Decisões estratégicas =========
    st.markdown("<div class='nx-block'>", unsafe_allow_html=True)
    st.markdown("### 🧭 Decisões estratégicas")
    bullets = []
    if "listed_in" in df_f.columns:
        _g = explode_column(df_f, "listed_in")
        if not _g.empty:
            topg = _g["listed_in"].value_counts().head(3).index.tolist()
            if topg: bullets.append(f"🔍 **Gêneros líderes**: {', '.join(topg)} — priorizar licenciamento/destaque editorial.")
    if "country" in df_f.columns:
        _c = explode_column(df_f, "country")
        if not _c.empty:
            topc = _c["country"].value_counts().head(3).index.tolist()
            if topc: bullets.append(f"🌐 **Praças prioritárias**: {', '.join(topc)} — campanhas locais e bundles.")
    if "release_year" in df_f.columns and df_f["release_year"].notna().any():
        yr = df_f.groupby("release_year").size().reset_index(name="qtd").sort_values("release_year")
        if not yr.empty:
            trend = "crescimento recente" if yr["qtd"].tail(3).is_monotonic_increasing else "variação nos últimos anos"
            bullets.append(f"📈 **Lançamentos**: {trend} — alinhar aquisições ao calendário.")
    if "score" in df_f.columns and df_f["score"].notna().any():
        bullets.append(f"⭐ **Qualidade média**: {df_f['score'].mean():.1f} — revisar long tail de baixo score.")

    if bullets:
        for b in bullets: st.markdown(f"- {b}")
    else:
        st.markdown("- Ajuste os filtros para revelar **prioridades de licenciamento** e **campanhas regionais**.")
    st.caption("Dica: salve visões filtradas como presets para reuniões executivas.")
    st.markdown("</div>", unsafe_allow_html=True)  # fecha .nx-wrap


