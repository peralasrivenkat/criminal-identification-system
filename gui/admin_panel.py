from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .ui_theme import (
        FONTS,
        PALETTE,
        ScrollableContent,
        configure_window,
        create_icon_button,
        create_page_header,
        create_panel,
        create_primary_button,
        create_secondary_button,
        render_preview,
        set_grid_weights,
        style_entry,
        style_treeview,
    )
except ImportError:  # pragma: no cover
    from ui_theme import (
        FONTS,
        PALETTE,
        ScrollableContent,
        configure_window,
        create_icon_button,
        create_page_header,
        create_panel,
        create_primary_button,
        create_secondary_button,
        render_preview,
        set_grid_weights,
        style_entry,
        style_treeview,
    )

from config import REGISTRATION_IMAGE_COUNT
from database.db_operations import get_all_criminals, register_criminal_with_images


class AdminPanel(tk.Frame):
    def __init__(self, parent: tk.Misc, *, on_back=None) -> None:
        super().__init__(parent, bg=PALETTE["bg"])
        self.parent = parent
        self.on_back = on_back
        self.image_paths: list[Path] = []
        self.current_image_index = 0
        self.entries: dict[str, tk.Entry] = {}

        self.preview_label: tk.Label | None = None
        self.preview_counter: tk.Label | None = None
        self.preview_slider: tk.Scale | None = None
        self.summary_label: tk.Label | None = None
        self.tree: ttk.Treeview | None = None

        self._build_ui()
        self.refresh_criminals()
        self._update_preview()

    def _build_ui(self) -> None:
        shell = tk.Frame(self, bg=PALETTE["bg"], padx=24, pady=24)
        shell.pack(fill="both", expand=True)

        create_page_header(shell, on_back=self.on_back, logo_size=(600, 144))

        content = tk.Frame(shell, bg=PALETTE["bg"])
        content.pack(fill="both", expand=True)
        set_grid_weights(content, columns=(0, 1), rows=(0,))

        form_panel = create_panel(content, padx=22, pady=22)
        form_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        sidebar_scroll = ScrollableContent(content, bg=PALETTE["bg"])
        sidebar_scroll.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        sidebar = tk.Frame(sidebar_scroll.content, bg=PALETTE["bg"])
        sidebar.pack(fill="both", expand=True)

        preview_panel = create_panel(sidebar, padx=22, pady=22)
        preview_panel.pack(fill="x", pady=(0, 10))

        records_panel = create_panel(sidebar, padx=22, pady=22)
        records_panel.pack(fill="both", expand=True, pady=(10, 0))

        tk.Label(
            form_panel,
            text="Criminal Details",
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            font=FONTS["title"],
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(
            form_panel,
            text="Use at least 5 clear source images so the saved aligned faces are strong enough for training.",
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            wraplength=620,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 14))

        field_defs = [
            ("name", "Name"),
            ("dob", "Dob"),
            ("moles", "Moles"),
            ("nationality", "Nationality"),
            ("region", "Region"),
            ("crime", "Crime"),
            ("num_crimes", "Num Crimes"),
        ]
        for index, (field_key, field_label) in enumerate(field_defs, start=2):
            tk.Label(
                form_panel,
                text=field_label,
                bg=PALETTE["card"],
                fg=PALETTE["text"],
                font=FONTS["body_bold"],
            ).grid(row=index, column=0, sticky="w", pady=4, padx=(0, 12))
            entry = tk.Entry(form_panel)
            style_entry(entry)
            entry.grid(row=index, column=1, sticky="ew", pady=4)
            self.entries[field_key] = entry

        form_panel.grid_columnconfigure(1, weight=1)

        button_row = tk.Frame(form_panel, bg=PALETTE["card"])
        button_row.grid(row=len(field_defs) + 2, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        set_grid_weights(button_row, columns=(0, 1))
        create_primary_button(button_row, text="Upload Images", command=self.upload_images).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        create_primary_button(button_row, text="Register Criminal", command=self.save_criminal).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )

        create_secondary_button(form_panel, text="Refresh Records", command=self.reset_form).grid(
            row=len(field_defs) + 3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 2),
        )

        tk.Label(
            preview_panel,
            text="Selected Images",
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            font=FONTS["title"],
        ).pack(anchor="w")
        tk.Label(
            preview_panel,
            text="Review the uploaded images here before registration.",
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(10, 16))

        self.preview_label = tk.Label(preview_panel)
        self.preview_label.pack(fill="x")

        nav_row = tk.Frame(preview_panel, bg=PALETTE["card"], pady=14)
        nav_row.pack(fill="x")
        nav_row.grid_columnconfigure(1, weight=1)
        create_icon_button(nav_row, image_name="previous.png", command=self.show_previous_image, bg=PALETTE["card"]).grid(
            row=0, column=0, sticky="w"
        )
        self.preview_counter = tk.Label(
            nav_row,
            text="No images selected",
            bg=PALETTE["card"],
            fg=PALETTE["accent"],
            font=FONTS["section"],
        )
        self.preview_counter.grid(row=0, column=1)
        create_icon_button(nav_row, image_name="next.png", command=self.show_next_image, bg=PALETTE["card"]).grid(
            row=0, column=2, sticky="e"
        )

        self.preview_slider = tk.Scale(
            preview_panel,
            from_=1,
            to=1,
            orient="horizontal",
            showvalue=0,
            highlightthickness=0,
            troughcolor=PALETTE["surface_alt"],
            bg=PALETTE["card"],
            activebackground=PALETTE["accent"],
            command=self._on_slider_change,
        )
        self.preview_slider.pack(fill="x")

        self.summary_label = tk.Label(
            preview_panel,
            text="Recommended dataset: at least 5 source images.",
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            justify="left",
        )
        self.summary_label.pack(anchor="w", pady=(14, 0))

        records_header = tk.Frame(records_panel, bg=PALETTE["card"])
        records_header.pack(fill="x")
        tk.Label(
            records_header,
            text="Registered Criminals",
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            font=FONTS["title"],
        ).pack(side="left")

        tk.Label(
            records_panel,
            text="Current records stored in the database.",
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
        ).pack(anchor="w", pady=(10, 14))

        table_frame = tk.Frame(records_panel, bg=PALETTE["card"])
        table_frame.pack(fill="both", expand=True)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        style_name = style_treeview()
        self.tree = ttk.Treeview(
            table_frame,
            columns=("id", "name", "crime"),
            show="headings",
            style=style_name,
            height=9,
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("crime", text="Crime")
        self.tree.column("id", width=70, anchor="center")
        self.tree.column("name", width=170, anchor="w")
        self.tree.column("crime", width=170, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        tree_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

    def upload_images(self) -> None:
        files = filedialog.askopenfilenames(
            title=f"Select {REGISTRATION_IMAGE_COUNT} Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not files:
            return

        if len(files) != REGISTRATION_IMAGE_COUNT:
            messagebox.showerror("Image Count Error", f"Please select exactly {REGISTRATION_IMAGE_COUNT} images.")
            return

        self.image_paths = [Path(path) for path in files]
        self.current_image_index = 0
        self._update_preview()

    def save_criminal(self) -> None:
        try:
            name = self.entries["name"].get().strip()
            dob = self.entries["dob"].get().strip() or None
            moles = self.entries["moles"].get().strip()
            nationality = self.entries["nationality"].get().strip()
            region = self.entries["region"].get().strip()
            crime = self.entries["crime"].get().strip()
            num_crimes_text = self.entries["num_crimes"].get().strip()

            if not name:
                raise ValueError("Name is required.")
            if not num_crimes_text:
                raise ValueError("Num Crimes is required.")
            if len(self.image_paths) != REGISTRATION_IMAGE_COUNT:
                raise ValueError(f"Please upload exactly {REGISTRATION_IMAGE_COUNT} images before registration.")

            num_crimes = int(num_crimes_text)
            criminal_id, stored_paths = register_criminal_with_images(
                name=name,
                dob=dob,
                moles=moles,
                nationality=nationality,
                region=region,
                crime=crime,
                num_crimes=num_crimes,
                image_paths=[str(path) for path in self.image_paths],
            )
            messagebox.showinfo(
                "Registration Complete",
                f"Criminal {criminal_id} registered successfully.\nSaved {len(stored_paths)} aligned training images.",
            )
            self.reset_form()
        except ValueError as error:
            messagebox.showerror("Registration Error", str(error))
        except Exception as error:
            messagebox.showerror("Save Error", str(error))

    def refresh_criminals(self) -> None:
        if self.tree is None:
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
        for criminal in get_all_criminals():
            self.tree.insert(
                "",
                "end",
                values=(
                    criminal.get("id", ""),
                    criminal.get("name", ""),
                    criminal.get("crime", ""),
                ),
            )

    def reset_form(self) -> None:
        for entry in self.entries.values():
            entry.delete(0, "end")
        self.image_paths = []
        self.current_image_index = 0
        self._update_preview()
        self.refresh_criminals()

    def show_previous_image(self) -> None:
        if not self.image_paths:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self._update_preview()

    def show_next_image(self) -> None:
        if not self.image_paths:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self._update_preview()

    def _on_slider_change(self, value: str) -> None:
        if not self.image_paths:
            return
        self.current_image_index = max(0, min(len(self.image_paths) - 1, int(float(value)) - 1))
        self._update_preview(update_slider=False)

    def _update_preview(self, *, update_slider: bool = True) -> None:
        if self.preview_label is None or self.preview_counter is None or self.preview_slider is None or self.summary_label is None:
            return

        if not self.image_paths:
            render_preview(
                self.preview_label,
                path=None,
                max_size=(420, 300),
                empty_text="Upload criminal images to preview them here.",
            )
            self.preview_counter.configure(text="No images selected")
            self.preview_slider.configure(from_=1, to=1)
            self.preview_slider.set(1)
            self.summary_label.configure(text=f"Recommended dataset: at least {REGISTRATION_IMAGE_COUNT} source images.")
            return

        current_path = self.image_paths[self.current_image_index]
        render_preview(self.preview_label, path=current_path, max_size=(420, 300))
        self.preview_counter.configure(text=f"Image {self.current_image_index + 1} of {len(self.image_paths)}")
        self.preview_slider.configure(from_=1, to=len(self.image_paths))
        if update_slider:
            self.preview_slider.set(self.current_image_index + 1)
        self.summary_label.configure(text=f"Selected {len(self.image_paths)} source image(s).")


def launch_admin_panel() -> None:
    root = tk.Tk()
    configure_window(root, title="Criminal Identification System", geometry="1320x900")
    panel = AdminPanel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    launch_admin_panel()
