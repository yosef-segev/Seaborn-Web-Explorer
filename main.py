from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd

from services.data_service import DataService
from services.analysis_service import AnalysisService

app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Services (load dataset once on startup)
data_store = DataService("titanic")
analyzer = AnalysisService(data_store.get_df())


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "dataset_name": data_store.dataset_name,

    })


@app.get("/questions")
def questions_page(request: Request):
    return templates.TemplateResponse("questions.html", {
        "request": request,
        "instruction": "Please select a question from the menu to start the analysis.",
        "question_id": None
    })


@app.get("/questions/{question_id}")
def run_specific_question(question_id: int, request: Request):
    title, result_html, plot_filename = analyzer.run_question(question_id)
    return templates.TemplateResponse("questions.html", {
        "request": request,
        "title": title,
        "result": result_html,
        "plot_url": plot_filename,
        "question_id": question_id,

    })


@app.get("/data")
def data_page(request: Request):
    # Show the form page
    return templates.TemplateResponse("data.html", {"request": request})


@app.post("/data")
def handle_data_request(
    request: Request,
    columns: str = Form(""),
    filter_col: str = Form(""),
    op: str = Form("=="),
    value: str = Form(""),
    limit: int = Form(20),
):
    df = data_store.get_df()

    # case-insensitive column map
    col_map = {c.lower(): c for c in df.columns}

    # 1) Columns
    default_cols = ["survived", "class", "sex", "age", "fare", "embarked"]
    default_cols = [c for c in default_cols if c in df.columns]

    columns = (columns or "").strip()
    if columns:
        requested_cols = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in requested_cols if c not in df.columns]
        if missing:
            return templates.TemplateResponse("data.html", {
                "request": request,
                "error": f"Column(s) not found: {', '.join(missing)}",
                "columns": columns,
                "filter_col": filter_col,
                "op": op,
                "value": value,
                "limit": limit,
            })
        df_view = df[requested_cols].copy()
    else:
        df_view = df[default_cols].copy()

    # 2) Filter
    filter_col = (filter_col or "").strip()
    value = (value or "").strip()

    if filter_col and value:
        fkey = filter_col.lower()
        if fkey not in col_map:
            return templates.TemplateResponse("data.html", {
                "request": request,
                "error": f"Filter column not found: {filter_col}",
                "columns": columns,
                "filter_col": filter_col,
                "op": op,
                "value": value,
                "limit": limit,
            })

        real_col = col_map[fkey]
        s = df[real_col]

        try:
            if op == "contains":
                mask = s.astype(str).str.contains(value, case=False, na=False)


            elif op in ("==", "!="):
                left_num = pd.to_numeric(s, errors="coerce")
                right_num = pd.to_numeric(value, errors="coerce")
                if pd.notna(right_num) and left_num.notna().any():

                    # numeric equality
                    if op == "==":
                        mask = left_num == float(right_num)
                    else:
                        mask = left_num != float(right_num)
                else:  # string equality (case-insensitive)
                    left = s.astype(str).str.strip().str.lower()
                    right = value.strip().lower()
                    if op == "==":
                        mask = left == right
                    else:
                        mask = left != right


            else:
                num = float(value)
                left_num = pd.to_numeric(s, errors="coerce")
                if op == ">":
                    mask = left_num > num
                elif op == "<":
                    mask = left_num < num
                elif op == ">=":
                    mask = left_num >= num
                elif op == "<=":
                    mask = left_num <= num
                else:
                    return templates.TemplateResponse("data.html", {
                        "request": request,
                        "error": f"Invalid operator: {op}",
                        "columns": columns,
                        "filter_col": filter_col,
                        "op": op,
                        "value": value,
                        "limit": limit,
                    })

            df_view = df.loc[mask, df_view.columns].copy()

        except Exception:
            return templates.TemplateResponse("data.html", {
                "request": request,
                "error": "Invalid filter (operator/value mismatch).",
                "columns": columns,
                "filter_col": filter_col,
                "op": op,
                "value": value,
                "limit": limit,
            })

    # 3) Limit
    try:
        limit = int(limit)
    except Exception:
        limit = 20

    limit = max(1, limit)
    df_view = df_view.head(limit).copy()

    # 4) Rows
    df_view.insert(0, "Rows", range(1, len(df_view) + 1))

    if df_view.empty:
        return templates.TemplateResponse("data.html", {
            "request": request,
            "error": "No rows matched your filter. Try a different value/operator.",
            "columns": columns,
            "filter_col": filter_col,
            "op": op,
            "value": value,
            "limit": limit,
        })

    table_html = df_view.to_html(
        index=False,
        classes="table table-striped table-hover align-middle mb-0"
    )

    return templates.TemplateResponse("data.html", {
        "request": request,
        "table": table_html,
        # keep form values
        "columns": columns,
        "filter_col": filter_col,
        "op": op,
        "value": value,
        "limit": limit,
    })
