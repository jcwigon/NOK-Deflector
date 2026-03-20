# -*- coding: utf-8 -*-
import base64
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from side_model import SideClassifier, draw_overlay_bgr

st.set_page_config(page_title="Side check (RIGHT vs LEFT)", layout="wide")
st.title("Kontrola strony montażu: RIGHT vs LEFT")

MODEL_DIR_DEFAULT = Path("models/number_side")


@st.cache_resource
def load_model(model_dir: str):
    return SideClassifier.load(model_dir)


# --- session defaults ---
if "selected_row_id" not in st.session_state:
    st.session_state["selected_row_id"] = None
if "picked_from_image" not in st.session_state:
    st.session_state["picked_from_image"] = False


with st.sidebar:
    st.header("Ustawienia")
    model_dir = st.text_input("Model dir", value=str(MODEL_DIR_DEFAULT))
    expected_side = st.selectbox("Oczekiwana strona (kafel)", ["RIGHT", "LEFT"], index=0)
    show_ok = st.checkbox("Pokazuj też OK", value=False)

clf = load_model(model_dir)

st.write(f"**Expected side:** `{expected_side}` | **thr:** `{clf.cfg.thr}` | **model_dir:** `{model_dir}`")

uploads = st.file_uploader(
    "Wgraj zdjęcia (mogą być pomieszane):",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Wgraj pliki aby zobaczyć wyniki.")
    st.stop()

rows = []
views = []

for uf in uploads:
    data = np.frombuffer(uf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.warning(f"Nie mogę odczytać: {uf.name}")
        continue

    pred = clf.predict_from_gray(img)
    pred["name"] = uf.name
    res = clf.check_expected(pred, expected_side=expected_side)

    # id do powiązania wiersz <-> zdjęcie
    res["row_id"] = len(rows)

    rows.append(res)
    views.append((uf.name, img, res))

df = pd.DataFrame(rows)

# ------------------ PODSUMOWANIE ------------------
st.subheader("Podsumowanie")

counts = df["status"].value_counts(dropna=False).to_dict()
wrong = int(counts.get("WRONG_SIDE", 0))
ok = int(counts.get("OK", 0))
unc = int(counts.get("UNCERTAIN", 0))

c1, c2, c3 = st.columns(3)


def summary_tile(title: str, value: int, accent: str):
    st.markdown(
        f"""
        <div style="border: 1px solid #eaeaea; border-radius: 12px; padding: 14px 16px; background: #ffffff;">
          <div style="font-size: 18px; font-weight: 800; margin-bottom: 6px; color: #111827; display:flex; align-items:center; gap:10px;">
            <span style="width:10px; height:10px; border-radius:999px; background:{accent}; display:inline-block;"></span>
            {title}
          </div>
          <div style="font-size: 44px; font-weight: 900; line-height: 1.0; color: #111827;">
            {value}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c1:
    summary_tile("Wrong side", wrong, "#d32f2f")
with c2:
    summary_tile("OK", ok, "#2e7d32")
with c3:
    summary_tile("Uncertain", unc, "#ef6c00")

# ------------------ TABELA (AgGrid) ------------------
st.subheader("Tabela (kliknij checkbox, żeby podświetlić obrazek niżej)")

# przycisk do odklikania (czyszczenia selekcji)
clear_col, _ = st.columns([0.25, 0.75])
with clear_col:
    if st.button("Wyczyść wybór"):
        st.session_state["selected_row_id"] = None
        st.session_state["picked_from_image"] = False
        st.rerun()

order = {"WRONG_SIDE": 0, "UNCERTAIN": 1, "OK": 2}
df["_status_order"] = df["status"].map(order).fillna(99).astype(int)

df_sorted = df.sort_values(["_status_order", "confidence"], ascending=[True, False]).reset_index(drop=True)
df_table = df_sorted.drop(columns=["_status_order"], errors="ignore").copy()

# AgGrid czasem ma problem z listą w komórce — zamień na string
if "box_xyxy" in df_table.columns:
    df_table["box_xyxy"] = df_table["box_xyxy"].astype(str)

gb = GridOptionsBuilder.from_dataframe(df_table)

# Checkbox selection (łatwiej też odznaczyć klikając checkbox ponownie — zależy od wersji aggrid,
# więc i tak mamy przycisk "Wyczyść wybór")
gb.configure_selection(selection_mode="single", use_checkbox=True)

gb.configure_pagination(enabled=False)  # scroll zamiast paginacji
gb.configure_default_column(resizable=True, sortable=True, filter=True)
grid_options = gb.build()
grid_options["rowHeight"] = 28

grid_resp = AgGrid(
    df_table,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=520,
    theme="streamlit",
)

selected_rows_obj = grid_resp.get("selected_rows", [])

# selected_rows może być listą dictów albo DataFrame
if isinstance(selected_rows_obj, pd.DataFrame):
    selected_rows = selected_rows_obj.to_dict(orient="records")
elif isinstance(selected_rows_obj, list):
    selected_rows = selected_rows_obj
else:
    selected_rows = []

# WAŻNE: jeżeli w tym rerunie kliknęliśmy "Wybierz" przy obrazku,
# to NIE NADPISUJEMY selected_row_id z gridu
if st.session_state.get("picked_from_image", False):
    selected_row_id = st.session_state.get("selected_row_id")
    # zdejmujemy flagę po jednym przebiegu
    st.session_state["picked_from_image"] = False
else:
    if len(selected_rows) > 0:
        selected_row_id = selected_rows[0].get("row_id")
        st.session_state["selected_row_id"] = selected_row_id
    else:
        selected_row_id = st.session_state.get("selected_row_id")

# ------------------ DOWNLOAD CSV + XLSX ------------------
st.subheader("Eksport")
col_csv, col_xlsx = st.columns(2)

df_out = df_table.copy()

with col_csv:
    st.download_button(
        "Pobierz CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"side_check_expected_{expected_side}.csv",
        mime="text/csv",
        use_container_width=True,
    )

xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
    df_out.to_excel(writer, index=False, sheet_name="results")

with col_xlsx:
    st.download_button(
        "Pobierz XLSX",
        data=xlsx_buf.getvalue(),
        file_name=f"side_check_expected_{expected_side}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ------------------ PODGLĄD ------------------
st.subheader("Podgląd z zaznaczeniem (WRONG_SIDE / UNCERTAIN + opcjonalnie OK)")
cols = st.columns(2)
i = 0

for name, img_gray, res in views:
    if (res["status"] == "OK") and (not show_ok):
        continue

    vis = draw_overlay_bgr(img_gray, res)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    is_selected = (selected_row_id is not None) and (res.get("row_id") == selected_row_id)

    with cols[i % 2]:
        top_left, top_right = st.columns([0.75, 0.25])
        with top_left:
            st.markdown(f"**{name}**")
        with top_right:
            if st.button("Wybierz", key=f"select_{res['row_id']}"):
                st.session_state["selected_row_id"] = res["row_id"]
                st.session_state["picked_from_image"] = True
                st.rerun()

        st.caption(
            f'{res["status"]} | expected={expected_side} | pred={res["side_pred"]} | conf={res["confidence"]:.3f}'
        )

        border = "4px solid rgba(59, 130, 246, 0.95)" if is_selected else "1px solid #eaeaea"
        shadow = "0 0 0 6px rgba(59, 130, 246, 0.25)" if is_selected else "none"

        ok_png, png = cv2.imencode(".png", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
        if not ok_png:
            st.error("Nie mogę zakodować podglądu do PNG")
        else:
            b64 = base64.b64encode(png.tobytes()).decode("utf-8")
            st.markdown(
                f"""
                <div style="border:{border}; box-shadow:{shadow}; border-radius:12px; padding:8px; margin-bottom:10px;">
                  <img src="data:image/png;base64,{b64}" style="width:100%; height:auto; border-radius:8px;" />
                </div>
                """,
                unsafe_allow_html=True,
            )

    i += 1

if i == 0:
    st.info("Brak błędów/niepewnych (lub wyłączone pokazywanie OK).")