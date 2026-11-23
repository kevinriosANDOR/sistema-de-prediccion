from __future__ import annotations

import math
from pathlib import Path
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:
    raise SystemExit(
        "Falta matplotlib. Instala las dependencias con: python -m pip install -r requirements.txt"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "jugadores.csv"
PREDICTION_CHART_DIR = BASE_DIR / "outputs" / "graficas_predicciones"

COLORS = {
    "bg": "#0b1220",
    "sidebar": "#111b2e",
    "panel": "#17243a",
    "panel_alt": "#1d2c46",
    "line": "#2b3d5c",
    "text": "#f4f7fb",
    "muted": "#93a5bf",
    "cyan": "#42d3c5",
    "blue": "#5b8def",
    "orange": "#ffb45b",
    "red": "#ff7182",
    "green": "#65d38b",
}

DATA_COLUMNS = [
    "Jugador",
    "Edad",
    "Valoracion",
    "Potencial",
    "Goles_Asistencias",
    "Valor_Mercado_M",
]

FEATURE_COLUMNS = ["Edad", "Valoracion", "Potencial", "Goles_Asistencias"]
FEATURE_LABELS = {
    "Edad": "Edad",
    "Valoracion": "Valoración general",
    "Potencial": "Potencial",
    "Goles_Asistencias": "Goles + asistencias",
}


class FichajesApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MercadoFútbol | Modelo de valor de fichajes")
        self.root.geometry("1260x780")
        self.root.minsize(1080, 680)
        self.root.configure(bg=COLORS["bg"])

        self.data = pd.DataFrame()
        self.beta: Optional[np.ndarray] = None
        self.metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        self.predictions = np.array([])
        self.current_fig: Optional[Figure] = None
        self.prediction_fig: Optional[Figure] = None
        self.prediction_chart_path: Optional[Path] = None
        self.current_chart_kind = "Valor real vs. estimado"
        self.data_source = str(DEFAULT_DATASET)
        self.nav_buttons: dict[str, tk.Button] = {}
        self.prediction_vars: dict[str, tk.StringVar] = {}
        self.prediction_result: Optional[tk.Label] = None
        self._configure_styles()
        self.show_login()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["panel"],
            borderwidth=0,
            rowheight=32,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["panel_alt"],
            foreground=COLORS["muted"],
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
        )
        style.map("Treeview", background=[("selected", "#284c73")])
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["panel_alt"],
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["cyan"],
        )

    def _clear_root(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def show_login(self) -> None:
        self._clear_root()
        self.root.bind("<Return>", lambda _event: self._login())

        outer = tk.Frame(self.root, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True)

        accent = tk.Frame(outer, bg=COLORS["cyan"], width=7)
        accent.pack(side="left", fill="y")

        card = tk.Frame(outer, bg=COLORS["panel"], padx=54, pady=48)
        card.place(relx=0.5, rely=0.5, anchor="center", width=480, height=535)

        tk.Label(
            card,
            text="MF",
            bg=COLORS["cyan"],
            fg=COLORS["bg"],
            font=("Segoe UI Black", 23),
            width=3,
            height=1,
        ).pack(anchor="w")
        tk.Label(
            card,
            text="MercadoFútbol",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 26),
        ).pack(anchor="w", pady=(22, 5))
        tk.Label(
            card,
            text="Sistema visual para estimar el valor de fichajes",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(0, 34))

        tk.Label(card, text="USUARIO", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI Semibold", 9)).pack(anchor="w")
        self.username_entry = self._entry(card, "admin")
        self.username_entry.pack(fill="x", pady=(7, 20))

        tk.Label(card, text="CONTRASEÑA", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI Semibold", 9)).pack(anchor="w")
        self.password_entry = self._entry(card, "futbol2025", show="•")
        self.password_entry.pack(fill="x", pady=(7, 27))

        tk.Button(
            card,
            text="INICIAR SESIÓN",
            command=self._login,
            bg=COLORS["cyan"],
            fg=COLORS["bg"],
            activebackground="#69e6da",
            activeforeground=COLORS["bg"],
            relief="flat",
            cursor="hand2",
            font=("Segoe UI Semibold", 11),
            padx=14,
            pady=12,
        ).pack(fill="x")

        tk.Label(
            card,
            text="Acceso de demostración: admin / futbol2025",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(26, 0))

    def _entry(self, parent: tk.Widget, value: str = "", show: str = "") -> tk.Entry:
        entry = tk.Entry(
            parent,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            insertbackground=COLORS["cyan"],
            relief="flat",
            font=("Segoe UI", 12),
            show=show,
        )
        entry.insert(0, value)
        entry.configure(highlightthickness=1, highlightbackground=COLORS["line"], highlightcolor=COLORS["cyan"])
        return entry

    def _login(self) -> None:
        if self.username_entry.get().strip() == "admin" and self.password_entry.get() == "futbol2025":
            self.root.unbind("<Return>")
            self.show_dashboard()
        else:
            messagebox.showerror("Acceso denegado", "Usuario o contraseña incorrectos.")

    def show_dashboard(self) -> None:
        self._clear_root()
        self._read_dataset(DEFAULT_DATASET)

        self.sidebar = tk.Frame(self.root, bg=COLORS["sidebar"], width=240)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = tk.Frame(self.root, bg=COLORS["bg"])
        self.content.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_header()
        self.main_view = tk.Frame(self.content, bg=COLORS["bg"])
        self.main_view.pack(fill="both", expand=True, padx=28, pady=(14, 28))
        self.show_view("summary")

    def _build_sidebar(self) -> None:
        brand = tk.Frame(self.sidebar, bg=COLORS["sidebar"], padx=25, pady=25)
        brand.pack(fill="x")
        tk.Label(brand, text="MF", bg=COLORS["cyan"], fg=COLORS["bg"], font=("Segoe UI Black", 18), width=3).pack(anchor="w")
        tk.Label(brand, text="MercadoFútbol", bg=COLORS["sidebar"], fg=COLORS["text"], font=("Segoe UI Semibold", 17)).pack(anchor="w", pady=(13, 0))
        tk.Label(brand, text="ANÁLISIS DE FICHAJES", bg=COLORS["sidebar"], fg=COLORS["muted"], font=("Segoe UI Semibold", 8)).pack(anchor="w", pady=(3, 0))

        tk.Label(self.sidebar, text="NAVEGACIÓN", bg=COLORS["sidebar"], fg=COLORS["muted"], font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=25, pady=(28, 9))
        items = [("summary", "Resumen", "▦"), ("players", "Jugadores", "♙"), ("prediction", "Predicción", "↗"), ("model", "Modelo lineal", "β")]
        items.insert(3, ("charts", "Gráficas", "▤"))
        for key, label, icon in items:
            button = tk.Button(
                self.sidebar,
                text=f"  {icon}   {label}",
                command=lambda view=key: self.show_view(view),
                anchor="w",
                bg=COLORS["sidebar"],
                fg=COLORS["muted"],
                activebackground=COLORS["panel_alt"],
                activeforeground=COLORS["text"],
                relief="flat",
                bd=0,
                padx=18,
                pady=13,
                font=("Segoe UI Semibold", 11),
                cursor="hand2",
            )
            button.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[key] = button

        tk.Frame(self.sidebar, bg=COLORS["sidebar"]).pack(fill="both", expand=True)
        tk.Button(
            self.sidebar,
            text="  ↪   Cerrar sesión",
            command=self.show_login,
            anchor="w",
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            activebackground=COLORS["panel_alt"],
            activeforeground=COLORS["text"],
            relief="flat",
            padx=18,
            pady=18,
            font=("Segoe UI", 10),
            cursor="hand2",
        ).pack(fill="x", padx=12, pady=(0, 15))

    def _build_header(self) -> None:
        header = tk.Frame(self.content, bg=COLORS["bg"], height=78)
        header.pack(fill="x", padx=28, pady=(18, 0))
        header.pack_propagate(False)
        self.page_title = tk.Label(header, text="Resumen del mercado", bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI Semibold", 23))
        self.page_title.pack(side="left", anchor="center")
        right = tk.Frame(header, bg=COLORS["bg"])
        right.pack(side="right", fill="y")
        tk.Button(
            right,
            text="▣  Guardar gráfica",
            command=self.save_current_chart,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            activebackground=COLORS["line"],
            activeforeground=COLORS["text"],
            relief="flat",
            padx=15,
            pady=9,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        ).pack(side="left", padx=(0, 18), pady=17)
        user = tk.Frame(right, bg=COLORS["bg"])
        user.pack(side="left", pady=12)
        tk.Label(user, text="Administrador", bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI Semibold", 10)).pack(anchor="e")
        tk.Label(user, text="Sesión activa", bg=COLORS["bg"], fg=COLORS["green"], font=("Segoe UI", 9)).pack(anchor="e")

    def show_view(self, view: str) -> None:
        if not hasattr(self, "main_view"):
            return
        for child in self.main_view.winfo_children():
            child.destroy()
        for key, button in self.nav_buttons.items():
            active = key == view
            button.configure(bg=COLORS["panel_alt"] if active else COLORS["sidebar"], fg=COLORS["text"] if active else COLORS["muted"])

        titles = {"summary": "Resumen del mercado", "players": "Base de jugadores", "prediction": "Predicción de fichaje", "model": "Modelo lineal múltiple"}
        titles["charts"] = "Gráficas de predicción"
        self.page_title.configure(text=titles[view])
        if view == "summary":
            self._build_summary_view()
        elif view == "players":
            self._build_players_view()
        elif view == "prediction":
            self._build_prediction_view()
        elif view == "charts":
            self._build_charts_view()
        else:
            self._build_model_view()

    def _read_dataset(self, path: Path) -> bool:
        try:
            data = pd.read_csv(path, encoding="utf-8-sig")
            missing = [column for column in DATA_COLUMNS if column not in data.columns]
            if missing:
                raise ValueError(f"Faltan columnas: {', '.join(missing)}")
            data = data[DATA_COLUMNS].copy()
            for column in FEATURE_COLUMNS + ["Valor_Mercado_M"]:
                data[column] = pd.to_numeric(data[column], errors="coerce")
            data = data.dropna(subset=FEATURE_COLUMNS + ["Valor_Mercado_M"])
            if len(data) < 5:
                raise ValueError("El CSV debe tener al menos 5 jugadores válidos.")
            self.data = data
            self.prediction_fig = None
            self.prediction_chart_path = None
            self.data_source = str(path)
            self._train_model()
            return True
        except Exception as exc:
            messagebox.showerror("CSV no válido", str(exc))
            return False

    def _train_model(self) -> None:
        feature_matrix = self.data[FEATURE_COLUMNS].to_numpy(dtype=float)
        target = self.data["Valor_Mercado_M"].to_numpy(dtype=float)
        design_matrix = np.column_stack([np.ones(len(feature_matrix)), feature_matrix])
        self.beta = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ target
        self.predictions = design_matrix @ self.beta
        errors = target - self.predictions
        self.metrics["mae"] = float(np.mean(np.abs(errors)))
        self.metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))
        total_variance = float(np.sum((target - np.mean(target)) ** 2))
        self.metrics["r2"] = float(1 - np.sum(errors**2) / total_variance) if total_variance else 0.0

    def _card(self, parent: tk.Widget, width: Optional[int] = None) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLORS["panel"], padx=18, pady=17)
        if width:
            frame.configure(width=width)
            frame.pack_propagate(False)
        return frame

    def _stat_card(self, parent: tk.Widget, label: str, value: str, accent: str) -> tk.Frame:
        card = self._card(parent)
        top = tk.Frame(card, bg=COLORS["panel"])
        top.pack(fill="x")
        tk.Label(top, text="●", bg=COLORS["panel"], fg=accent, font=("Segoe UI", 12)).pack(side="left")
        tk.Label(top, text=label.upper(), bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI Semibold", 9)).pack(side="left", padx=7)
        tk.Label(card, text=value, bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 23)).pack(anchor="w", pady=(15, 0))
        return card

    def _build_summary_view(self) -> None:
        stats = tk.Frame(self.main_view, bg=COLORS["bg"])
        stats.pack(fill="x", pady=(0, 18))
        stats.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="stats")
        cards = [
            ("Jugadores analizados", f"{len(self.data):,}", COLORS["cyan"]),
            ("Valor medio", f"€ {self.data['Valor_Mercado_M'].mean():.1f} M", COLORS["orange"]),
            ("Precisión R²", f"{self.metrics['r2'] * 100:.1f}%", COLORS["blue"]),
            ("Error RMSE", f"€ {self.metrics['rmse']:.2f} M", COLORS["red"]),
        ]
        for index, (label, value, accent) in enumerate(cards):
            card = self._stat_card(stats, label, value, accent)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 7, 0 if index == 3 else 7))

        chart_card = self._card(self.main_view)
        chart_card.pack(fill="both", expand=True)
        title_row = tk.Frame(chart_card, bg=COLORS["panel"])
        title_row.pack(fill="x", pady=(0, 10))
        tk.Label(title_row, text="Lectura del modelo", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 14)).pack(side="left")
        self.chart_selector = ttk.Combobox(title_row, state="readonly", values=["Valor real vs. estimado", "Valor vs. valoración", "Ranking de valor"], width=25)
        self.chart_selector.set(self.current_chart_kind)
        self.chart_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_chart(self.chart_selector.get()))
        self.chart_selector.pack(side="right")
        self.chart_host = tk.Frame(chart_card, bg=COLORS["panel"])
        self.chart_host.pack(fill="both", expand=True)
        self.root.after(50, lambda: self._refresh_chart(self.current_chart_kind))

    def _style_axis(self, axis) -> None:
        axis.tick_params(colors=COLORS["muted"], labelsize=9)
        for spine in axis.spines.values():
            spine.set_color(COLORS["line"])
        axis.grid(color=COLORS["line"], alpha=0.45, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.xaxis.label.set_color(COLORS["muted"])
        axis.yaxis.label.set_color(COLORS["muted"])

    def _refresh_chart(self, kind: str) -> None:
        if not hasattr(self, "chart_host"):
            return
        self.current_chart_kind = kind
        for child in self.chart_host.winfo_children():
            child.destroy()
        figure = Figure(figsize=(8.5, 4.1), dpi=100, facecolor=COLORS["panel"])
        axis = figure.add_subplot(111)
        axis.set_facecolor(COLORS["panel"])
        if kind == "Valor real vs. estimado":
            actual = self.data["Valor_Mercado_M"].to_numpy()
            axis.scatter(actual, self.predictions, color=COLORS["cyan"], s=60, alpha=0.85, edgecolors=COLORS["bg"], linewidth=0.8)
            limit = max(actual.max(), self.predictions.max()) * 1.08
            axis.plot([0, limit], [0, limit], color=COLORS["orange"], linestyle="--", linewidth=1.5, label="Estimación ideal")
            axis.set_xlabel("Valor real (€ M)")
            axis.set_ylabel("Valor estimado (€ M)")
            axis.set_title("Comparación entre valor real y predicho", color=COLORS["text"], loc="left", pad=12, fontsize=12, fontweight="bold")
            axis.legend(facecolor=COLORS["panel_alt"], edgecolor=COLORS["line"], labelcolor=COLORS["text"], fontsize=8)
        elif kind == "Valor vs. valoración":
            axis.scatter(self.data["Valoracion"], self.data["Valor_Mercado_M"], color=COLORS["blue"], s=55, alpha=0.85, edgecolors=COLORS["bg"], linewidth=0.8)
            axis.set_xlabel("Valoración general")
            axis.set_ylabel("Valor de mercado (€ M)")
            axis.set_title("Relación entre valoración y precio", color=COLORS["text"], loc="left", pad=12, fontsize=12, fontweight="bold")
        else:
            ranking = self.data.sort_values("Valor_Mercado_M", ascending=False).head(10).sort_values("Valor_Mercado_M")
            axis.barh(ranking["Jugador"], ranking["Valor_Mercado_M"], color=COLORS["cyan"])
            axis.set_xlabel("Valor de mercado (€ M)")
            axis.set_title("Top 10 jugadores por valor", color=COLORS["text"], loc="left", pad=12, fontsize=12, fontweight="bold")
        self._style_axis(axis)
        axis.title.set_color(COLORS["text"])
        figure.tight_layout(pad=2)
        self.current_fig = figure
        canvas = FigureCanvasTkAgg(figure, master=self.chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_players_view(self) -> None:
        toolbar = tk.Frame(self.main_view, bg=COLORS["bg"])
        toolbar.pack(fill="x", pady=(0, 14))
        tk.Label(toolbar, text=f"Fuente: {Path(self.data_source).name}", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 10)).pack(side="left")
        tk.Button(toolbar, text="＋  Cargar CSV", command=self.import_csv, bg=COLORS["cyan"], fg=COLORS["bg"], activebackground="#69e6da", relief="flat", padx=15, pady=8, font=("Segoe UI Semibold", 10), cursor="hand2").pack(side="right")

        card = self._card(self.main_view)
        card.pack(fill="both", expand=True)
        tk.Label(card, text="Jugadores utilizados en el entrenamiento", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 14)).pack(anchor="w", pady=(0, 14))
        table_frame = tk.Frame(card, bg=COLORS["panel"])
        table_frame.pack(fill="both", expand=True)
        columns = [("Jugador", 180), ("Edad", 75), ("Valoración", 110), ("Potencial", 100), ("G+A", 100), ("Valor (€ M)", 120)]
        tree = ttk.Treeview(table_frame, columns=[col[0] for col in columns], show="headings")
        for column, width in columns:
            tree.heading(column, text=column)
            tree.column(column, width=width, anchor="w" if column == "Jugador" else "center")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for _, row in self.data.iterrows():
            tree.insert("", "end", values=(row["Jugador"], f"{row['Edad']:.0f}", f"{row['Valoracion']:.0f}", f"{row['Potencial']:.0f}", f"{row['Goles_Asistencias']:.0f}", f"€ {row['Valor_Mercado_M']:.1f} M"))

    def _build_prediction_view(self) -> None:
        body = tk.Frame(self.main_view, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)
        form = self._card(body, width=440)
        form.pack(side="left", fill="y", padx=(0, 18))
        result = self._card(body)
        result.pack(side="left", fill="both", expand=True)
        tk.Label(form, text="Datos del jugador", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 15)).pack(anchor="w")
        tk.Label(form, text="Completa las variables y estima su valor de mercado.", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 10), wraplength=350, justify="left").pack(anchor="w", pady=(6, 24))
        fields = [("Nombre del jugador", "Jugador demo", "name"), ("Edad", "23", "Edad"), ("Valoración general (0-100)", "80", "Valoracion"), ("Potencial (0-100)", "86", "Potencial"), ("Goles + asistencias", "18", "Goles_Asistencias")]
        self.prediction_vars = {}
        for label, default, key in fields:
            tk.Label(form, text=label, bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(10, 6))
            variable = tk.StringVar(value=default)
            self.prediction_vars[key] = variable
            entry = tk.Entry(form, textvariable=variable, bg=COLORS["panel_alt"], fg=COLORS["text"], insertbackground=COLORS["cyan"], relief="flat", font=("Segoe UI", 11))
            entry.configure(highlightthickness=1, highlightbackground=COLORS["line"], highlightcolor=COLORS["cyan"])
            entry.pack(fill="x", ipady=7)
        tk.Button(form, text="CALCULAR VALOR", command=self.predict_player, bg=COLORS["cyan"], fg=COLORS["bg"], activebackground="#69e6da", relief="flat", padx=14, pady=11, font=("Segoe UI Semibold", 10), cursor="hand2").pack(fill="x", pady=(27, 0))

        tk.Label(result, text="Resultado de la estimación", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 15)).pack(anchor="w")
        tk.Label(result, text="La predicción se calcula usando la ecuación normal de regresión lineal múltiple.", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 33))
        result_box = tk.Frame(result, bg=COLORS["panel_alt"], padx=25, pady=24)
        result_box.pack(fill="x")
        tk.Label(result_box, text="VALOR ESTIMADO", bg=COLORS["panel_alt"], fg=COLORS["muted"], font=("Segoe UI Semibold", 10)).pack(anchor="w")
        self.prediction_result = tk.Label(result_box, text="€ -- M", bg=COLORS["panel_alt"], fg=COLORS["cyan"], font=("Segoe UI Semibold", 34))
        self.prediction_result.pack(anchor="w", pady=(9, 4))
        self.prediction_detail = tk.Label(result_box, text="Ingresa los datos para generar una estimación.", bg=COLORS["panel_alt"], fg=COLORS["muted"], font=("Segoe UI", 10))
        self.prediction_detail.pack(anchor="w")
        info = tk.Frame(result, bg=COLORS["panel"])
        info.pack(fill="x", pady=(28, 0))
        tk.Label(info, text="Interpretación", bg=COLORS["panel"], fg=COLORS["orange"], font=("Segoe UI Semibold", 11)).pack(anchor="w")
        tk.Label(info, text="El valor no es una tasación oficial: es una estimación académica basada en los patrones del CSV cargado.", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 10), wraplength=620, justify="left").pack(anchor="w", pady=(7, 0))

    def _build_charts_view(self) -> None:
        card = self._card(self.main_view)
        card.pack(fill="both", expand=True)
        tk.Label(card, text="Estimación frente al mercado", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 15)).pack(anchor="w")
        tk.Label(
            card,
            text="Cada predicción genera aquí una gráfica comparativa y la guarda automáticamente en outputs/graficas_predicciones.",
            bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(6, 14))
        self.prediction_chart_host = tk.Frame(card, bg=COLORS["panel"])
        self.prediction_chart_host.pack(fill="both", expand=True)
        self._display_prediction_chart()

    def _display_prediction_chart(self) -> None:
        if not hasattr(self, "prediction_chart_host") or not self.prediction_chart_host.winfo_exists():
            return
        for child in self.prediction_chart_host.winfo_children():
            child.destroy()
        if self.prediction_fig is None:
            tk.Label(
                self.prediction_chart_host,
                text="Aún no hay gráficas. Ve a Predicción, completa los datos y pulsa CALCULAR VALOR.",
                bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 11),
            ).pack(expand=True)
            return
        canvas = FigureCanvasTkAgg(self.prediction_fig, master=self.prediction_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def predict_player(self) -> None:
        try:
            values = [float(self.prediction_vars[column].get()) for column in FEATURE_COLUMNS]
            age, rating, potential, contributions = values
            if not 15 <= age <= 45 or not 0 <= rating <= 100 or not 0 <= potential <= 100 or contributions < 0:
                raise ValueError("Revisa los rangos: edad 15-45, valoración/potencial 0-100 y G+A positivo.")
            if self.beta is None:
                raise ValueError("No hay un modelo entrenado.")
            estimate = float(np.array([1, *values]) @ self.beta)
            estimate = max(0.0, estimate)
            name = self.prediction_vars["name"].get().strip() or "Jugador demo"
            self._create_prediction_chart(name, estimate)
            self.prediction_result.configure(text=f"€ {estimate:.2f} M")
            self.prediction_detail.configure(text=f"{name} · gráfica guardada en {self.prediction_chart_path.relative_to(BASE_DIR)}")
        except ValueError as exc:
            messagebox.showwarning("Datos incompletos", str(exc))

    def _create_prediction_chart(self, name: str, estimate: float) -> None:
        values = self.data["Valor_Mercado_M"].to_numpy(dtype=float)
        figure = Figure(figsize=(9.2, 5.0), dpi=100, facecolor=COLORS["panel"])
        axis = figure.add_subplot(111)
        axis.set_facecolor(COLORS["panel"])
        bins = min(12, max(5, int(np.sqrt(len(values)))))
        axis.hist(values, bins=bins, color=COLORS["blue"], alpha=0.78, edgecolor=COLORS["panel"], label="Jugadores del CSV")
        median = float(np.median(values))
        axis.axvline(median, color=COLORS["orange"], linestyle="--", linewidth=2, label=f"Mediana del mercado: € {median:.1f} M")
        axis.axvline(estimate, color=COLORS["cyan"], linewidth=2.5, label=f"{name}: € {estimate:.2f} M")
        axis.set_xlabel("Valor de mercado (€ M)")
        axis.set_ylabel("Número de jugadores")
        axis.set_title("Valor estimado comparado con el mercado", color=COLORS["text"], loc="left", pad=12, fontsize=12, fontweight="bold")
        axis.legend(facecolor=COLORS["panel_alt"], edgecolor=COLORS["line"], labelcolor=COLORS["text"], fontsize=9)
        self._style_axis(axis)
        figure.tight_layout(pad=2)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_") or "jugador"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        PREDICTION_CHART_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = PREDICTION_CHART_DIR / f"prediccion_{safe_name}_{timestamp}.png"
        figure.savefig(chart_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
        self.prediction_fig = figure
        self.prediction_chart_path = chart_path
        self.current_fig = figure
        self._display_prediction_chart()

    def _build_model_view(self) -> None:
        top = tk.Frame(self.main_view, bg=COLORS["bg"])
        top.pack(fill="x", pady=(0, 18))
        top.grid_columnconfigure((0, 1, 2), weight=1, uniform="metrics")
        metric_cards = [("MAE", f"€ {self.metrics['mae']:.2f} M", "Error absoluto medio"), ("RMSE", f"€ {self.metrics['rmse']:.2f} M", "Raíz del error cuadrático"), ("R²", f"{self.metrics['r2']:.4f}", "Varianza explicada")]
        for index, (label, value, description) in enumerate(metric_cards):
            card = self._card(top)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 7, 0 if index == 2 else 7))
            tk.Label(card, text=label, bg=COLORS["panel"], fg=COLORS["cyan"], font=("Segoe UI Semibold", 11)).pack(anchor="w")
            tk.Label(card, text=value, bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 22)).pack(anchor="w", pady=(8, 0))
            tk.Label(card, text=description, bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 9)).pack(anchor="w", pady=(3, 0))

        lower = tk.Frame(self.main_view, bg=COLORS["bg"])
        lower.pack(fill="both", expand=True)
        coefficients = self._card(lower, width=470)
        coefficients.pack(side="left", fill="y", padx=(0, 18))
        explanation = self._card(lower)
        explanation.pack(side="left", fill="both", expand=True)
        tk.Label(coefficients, text="Coeficientes β", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 14)).pack(anchor="w")
        tk.Label(coefficients, text="Calculados con la ecuación normal", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 17))
        labels = ["β₀ · Intercepto", *[f"β{i} · {FEATURE_LABELS[col]}" for i, col in enumerate(FEATURE_COLUMNS, start=1)]]
        for label, value in zip(labels, self.beta if self.beta is not None else []):
            row = tk.Frame(coefficients, bg=COLORS["panel_alt"], padx=12, pady=10)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=COLORS["panel_alt"], fg=COLORS["muted"], font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, text=f"{value:.5f}", bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Consolas", 10)).pack(side="right")
        tk.Label(explanation, text="Aplicación de Álgebra Lineal", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 14)).pack(anchor="w")
        text = (
            "La matriz X contiene una columna de unos para el intercepto y cuatro columnas con las variables del jugador: "
            "edad, valoración, potencial y goles + asistencias. El vector Y contiene el valor de mercado en millones de euros.\n\n"
            "El sistema calcula los parámetros mediante:\n\n"
            "β = (XᵀX)⁻¹XᵀY\n\n"
            "Para evitar problemas cuando XᵀX no es invertible, la implementación usa la pseudoinversa de Moore-Penrose, "
            "equivalente para este propósito y más estable numéricamente.\n\n"
            f"Dimensión de X: {len(self.data)} × 5   |   Variables predictoras: 4   |   Objetivo: Valor_Mercado_M"
        )
        tk.Label(explanation, text=text, bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 11), justify="left", anchor="nw", wraplength=650).pack(fill="both", expand=True, anchor="nw", pady=(18, 0))

    def import_csv(self) -> None:
        path = filedialog.askopenfilename(title="Seleccionar dataset CSV", filetypes=[("Archivo CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if path and self._read_dataset(Path(path)):
            self.show_view("players")
            messagebox.showinfo("CSV cargado", f"Se cargaron {len(self.data)} jugadores y el modelo fue entrenado.")

    def save_current_chart(self) -> None:
        if self.current_fig is None:
            messagebox.showinfo("Sin gráfica", "Abre el Resumen para generar una gráfica antes de guardarla.")
            return
        path = filedialog.asksaveasfilename(title="Guardar gráfica", defaultextension=".png", filetypes=[("Imagen PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")], initialfile="grafica_mercado_fichajes.png")
        if path:
            self.current_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=self.current_fig.get_facecolor())
            messagebox.showinfo("Gráfica guardada", f"La gráfica se guardó en:\n{path}")


def main() -> None:
    root = tk.Tk()
    FichajesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
