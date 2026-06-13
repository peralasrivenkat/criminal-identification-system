from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .admin_panel import AdminPanel
    from .app import CriminalApp
    from .ui_theme import (
        FONTS,
        PALETTE,
        configure_window,
        create_page_header,
        create_metric_card,
        create_panel,
        create_primary_button,
        create_secondary_button,
        set_grid_weights,
    )
except ImportError:  # pragma: no cover
    from admin_panel import AdminPanel
    from app import CriminalApp
    from ui_theme import (
        FONTS,
        PALETTE,
        configure_window,
        create_page_header,
        create_metric_card,
        create_panel,
        create_primary_button,
        create_secondary_button,
        set_grid_weights,
    )

from config import TRAINING_METADATA_MODEL
from database.db_operations import get_all_criminals, initialize_database


class HomePage(tk.Frame):
    def __init__(self, parent: tk.Misc, *, open_admin, open_app, refresh_home) -> None:
        super().__init__(parent, bg=PALETTE["bg"])
        self.open_admin = open_admin
        self.open_app = open_app
        self.refresh_home = refresh_home
        self._build_ui()

    def _build_ui(self) -> None:
        shell = tk.Frame(self, bg=PALETTE["bg"], padx=24, pady=24)
        shell.pack(fill="both", expand=True)

        create_page_header(shell, on_back=None, logo_size=(600, 144))

        metrics = self._load_metrics()
        metric_row = tk.Frame(shell, bg=PALETTE["bg"], pady=18)
        metric_row.pack(fill="x")
        create_metric_card(metric_row, label="Registered Criminals", value=str(metrics["criminals"])).pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        create_metric_card(metric_row, label="Training Samples", value=str(metrics["samples"])).pack(
            side="left", fill="x", expand=True, padx=10
        )
        create_metric_card(metric_row, label="Embedding Backend", value=metrics["backend"]).pack(
            side="left", fill="x", expand=True, padx=(10, 0)
        )

        actions = tk.Frame(shell, bg=PALETTE["bg"])
        actions.pack(fill="both", expand=True)
        set_grid_weights(actions, columns=(0, 1), rows=(0,))

        admin_card = self._build_action_card(
            actions,
            title="Admin Panel",
            description="Register criminal profiles, upload source images, and manage the training dataset.",
            primary_text="Open Admin Panel",
            primary_command=self.open_admin,
        )
        admin_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        app_card = self._build_action_card(
            actions,
            title="Identification App",
            description="Upload an image or video and run the current identification pipeline from the same dashboard.",
            primary_text="Open Identification App",
            primary_command=self.open_app,
        )
        app_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        footer = tk.Frame(shell, bg=PALETTE["bg"], pady=16)
        footer.pack(fill="x")
        create_secondary_button(footer, text="Refresh Dashboard", command=self.refresh_home).pack(anchor="w")

    def _build_action_card(
        self,
        parent: tk.Misc,
        *,
        title: str,
        description: str,
        primary_text: str,
        primary_command,
    ) -> tk.Frame:
        card = create_panel(parent, padx=26, pady=26)
        tk.Label(card, text=title, bg=PALETTE["card"], fg=PALETTE["text"], font=FONTS["title"]).pack(anchor="w")
        tk.Label(
            card,
            text=description,
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(12, 24))
        create_primary_button(card, text=primary_text, command=primary_command).pack(fill="x")
        return card

    def _load_metrics(self) -> dict[str, str]:
        criminals = get_all_criminals()
        metadata = {}
        if TRAINING_METADATA_MODEL.exists():
            try:
                metadata = json.loads(TRAINING_METADATA_MODEL.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        return {
            "criminals": str(len(criminals)),
            "samples": str(metadata.get("train_samples", 0)),
            "backend": str(metadata.get("embedding_backend", "Not trained")),
        }


class ApplicationShell:
    def __init__(self, root: tk.Tk) -> None:
        initialize_database()
        self.root = root
        configure_window(root, title="Criminal Identification System", geometry="1320x900")
        self.current_screen: tk.Widget | None = None
        self.show_home()

    def _swap(self, widget: tk.Widget) -> None:
        if self.current_screen is not None and self.current_screen.winfo_exists():
            self.current_screen.destroy()
        self.current_screen = widget
        self.current_screen.pack(fill="both", expand=True)

    def show_home(self) -> None:
        self._swap(
            HomePage(
                self.root,
                open_admin=self.show_admin,
                open_app=self.show_app,
                refresh_home=self.show_home,
            )
        )

    def show_admin(self) -> None:
        self._swap(AdminPanel(self.root, on_back=self.show_home))

    def show_app(self) -> None:
        self._swap(CriminalApp(self.root, on_back=self.show_home))


def launch_home() -> None:
    root = tk.Tk()
    ApplicationShell(root)
    root.mainloop()


if __name__ == "__main__":
    launch_home()
