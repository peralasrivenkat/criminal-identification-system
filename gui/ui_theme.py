from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Iterable
from collections import deque

from PIL import Image, ImageOps, ImageTk


GUI_DIR = Path(__file__).resolve().parent
ASSET_DIR = GUI_DIR / "assets"

PALETTE = {
    "bg": "#0F4030",
    "bg_soft": "#16523E",
    "hero": "#133E31",
    "hero_alt": "#0B231B",
    "surface": "#F4EEDF",
    "surface_alt": "#E7DFC9",
    "card": "#FCFAF4",
    "card_alt": "#F7F1E4",
    "text": "#173127",
    "text_muted": "#5C6F65",
    "accent": "#2E97D4",
    "accent_hover": "#197DB8",
    "accent_deep": "#0C6DA6",
    "success": "#2E8B57",
    "warning": "#D79827",
    "danger": "#C55A43",
    "border": "#D6CCB8",
}

FONTS = {
    "hero": ("Georgia", 28, "bold"),
    "title": ("Georgia", 21, "bold"),
    "section": ("Georgia", 15, "bold"),
    "body": ("Segoe UI", 11),
    "body_bold": ("Segoe UI Semibold", 11),
    "small": ("Segoe UI", 10),
    "button": ("Segoe UI Semibold", 11),
    "metric": ("Georgia", 20, "bold"),
    "tag": ("Segoe UI Semibold", 10),
}


class ScrollableContent(tk.Frame):
    def __init__(self, parent: tk.Misc, *, bg: str) -> None:
        super().__init__(parent, bg=bg)
        self._bg = bg
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.content = tk.Frame(self.canvas, bg=bg)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.content.bind("<Enter>", self._bind_mousewheel)
        self.content.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        if event.delta == 0:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def configure_window(window: tk.Misc, *, title: str, geometry: str) -> None:
    window.title(title)
    window.geometry(geometry)
    window.configure(bg=PALETTE["bg"])


def create_panel(
    parent: tk.Misc,
    *,
    bg: str | None = None,
    padx: int = 18,
    pady: int = 18,
    borderwidth: int = 1,
) -> tk.Frame:
    return tk.Frame(
        parent,
        bg=bg or PALETTE["card"],
        bd=borderwidth,
        highlightthickness=0,
        padx=padx,
        pady=pady,
    )


def style_entry(entry: tk.Entry) -> None:
    entry.configure(
        bg="white",
        fg=PALETTE["text"],
        insertbackground=PALETTE["text"],
        relief="flat",
        highlightthickness=1,
        highlightbackground=PALETTE["border"],
        highlightcolor=PALETTE["accent"],
        font=FONTS["body"],
    )


def create_primary_button(
    parent: tk.Misc,
    *,
    text: str,
    command,
    width: int | None = None,
) -> tk.Button:
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        bg=PALETTE["accent"],
        fg="white",
        activebackground=PALETTE["accent_hover"],
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        padx=16,
        pady=10,
        font=FONTS["button"],
    )
    _apply_hover(button, base=PALETTE["accent"], hover=PALETTE["accent_hover"])
    return button


def create_secondary_button(
    parent: tk.Misc,
    *,
    text: str,
    command,
    width: int | None = None,
) -> tk.Button:
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        bg=PALETTE["surface_alt"],
        fg=PALETTE["text"],
        activebackground=PALETTE["border"],
        activeforeground=PALETTE["text"],
        relief="flat",
        bd=0,
        cursor="hand2",
        padx=16,
        pady=10,
        font=FONTS["button"],
    )
    _apply_hover(button, base=PALETTE["surface_alt"], hover=PALETTE["border"])
    return button


def create_icon_button(
    parent: tk.Misc,
    *,
    image_name: str,
    command,
    size: tuple[int, int] = (42, 42),
    bg: str = PALETTE["hero"],
    hover: str = PALETTE["bg_soft"],
) -> tk.Button:
    photo = load_asset_photo(image_name, size=size)
    button = tk.Button(
        parent,
        image=photo,
        command=command,
        bg=bg,
        activebackground=hover,
        relief="flat",
        bd=0,
        cursor="hand2",
        highlightthickness=0,
    )
    button.image = photo
    _apply_hover(button, base=bg, hover=hover)
    return button


def create_metric_card(
    parent: tk.Misc,
    *,
    label: str,
    value: str,
    tone: str = "card_alt",
) -> tk.Frame:
    panel = create_panel(parent, bg=PALETTE[tone], padx=18, pady=18, borderwidth=0)
    tk.Label(panel, text=label, bg=PALETTE[tone], fg=PALETTE["text_muted"], font=FONTS["small"]).pack(anchor="w")
    tk.Label(panel, text=value, bg=PALETTE[tone], fg=PALETTE["text"], font=FONTS["metric"]).pack(anchor="w", pady=(8, 0))
    return panel


def create_page_header(
    parent: tk.Misc,
    *,
    on_back=None,
    logo_size: tuple[int, int] = (500, 120),
    compact: bool = False,
) -> tk.Frame:
    header_pad = 16
    header_gap = 12
    logo_gap = 12
    side_slot = 68

    header = create_panel(parent, bg=PALETTE["hero"], padx=header_pad, pady=header_pad, borderwidth=0)
    header.pack(fill="x", pady=(0, header_gap))

    top_row = tk.Frame(header, bg=PALETTE["hero"])
    top_row.pack(fill="x")
    top_row.grid_columnconfigure(0, minsize=side_slot)
    top_row.grid_columnconfigure(1, weight=1)
    top_row.grid_columnconfigure(2, minsize=side_slot)

    if on_back is not None:
        create_icon_button(top_row, image_name="back.png", command=on_back, bg=PALETTE["hero"]).grid(
            row=0,
            column=0,
            sticky="w",
        )

    tk.Label(
        top_row,
        text="Criminal Identification System",
        bg=PALETTE["hero"],
        fg="white",
        font=FONTS["hero"],
    ).grid(row=0, column=1)

    logo_photo = load_asset_photo("logo.png", size=logo_size, make_black_transparent=True)
    logo_label = tk.Label(header, image=logo_photo, bg=PALETTE["hero"])
    logo_label.image = logo_photo
    logo_label.pack(anchor="center", pady=(logo_gap, 2))
    return header


def load_asset_photo(
    name: str,
    *,
    size: tuple[int, int] | None = None,
    make_black_transparent: bool = False,
) -> ImageTk.PhotoImage:
    path = ASSET_DIR / name
    with Image.open(path) as image:
        asset = ImageOps.exif_transpose(image).convert("RGBA")
        if make_black_transparent:
            asset = _make_border_black_transparent(asset)
        if size is not None:
            resample = getattr(Image, "Resampling", Image).LANCZOS
            asset = asset.resize(size, resample)
        return ImageTk.PhotoImage(asset)


def render_preview(
    label: tk.Label,
    *,
    path: str | Path | None = None,
    image: Image.Image | None = None,
    max_size: tuple[int, int] = (420, 320),
    empty_text: str = "No image selected",
) -> None:
    if path is None and image is None:
        label.configure(
            image="",
            text=empty_text,
            compound="center",
            bg=PALETTE["surface"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            padx=28,
            pady=84,
        )
        label.image = None
        return

    if image is None:
        with Image.open(path) as opened:
            preview_source = ImageOps.exif_transpose(opened).convert("RGB")
    else:
        preview_source = image.copy().convert("RGB")

    resample = getattr(Image, "Resampling", Image).LANCZOS
    preview_source.thumbnail(max_size, resample)
    photo = ImageTk.PhotoImage(preview_source)
    label.configure(image=photo, text="", bg=PALETTE["surface"], padx=0, pady=0)
    label.image = photo


def clear_children(parent: tk.Misc) -> None:
    for child in parent.winfo_children():
        child.destroy()


def set_grid_weights(widget: tk.Misc, *, columns: Iterable[int] = (), rows: Iterable[int] = ()) -> None:
    for column in columns:
        widget.grid_columnconfigure(column, weight=1)
    for row in rows:
        widget.grid_rowconfigure(row, weight=1)


def style_treeview(style_name: str = "Criminal.Treeview") -> str:
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        style_name,
        background="white",
        foreground=PALETTE["text"],
        fieldbackground="white",
        font=FONTS["small"],
        rowheight=28,
        borderwidth=0,
    )
    style.configure(
        "{0}.Heading".format(style_name),
        background=PALETTE["surface_alt"],
        foreground=PALETTE["text"],
        font=FONTS["body_bold"],
        relief="flat",
    )
    style.map(
        style_name,
        background=[("selected", PALETTE["accent"])],
        foreground=[("selected", "white")],
    )
    return style_name


def _apply_hover(button: tk.Button, *, base: str, hover: str) -> None:
    button.bind("<Enter>", lambda _event: button.configure(bg=hover))
    button.bind("<Leave>", lambda _event: button.configure(bg=base))


def _make_border_black_transparent(image: Image.Image) -> Image.Image:
    source = image.convert("RGBA")
    width, height = source.size
    pixels = source.load()
    visited = set()
    queue = deque()

    def is_background(x: int, y: int) -> bool:
        red, green, blue, alpha = pixels[x, y]
        return alpha > 0 and red < 20 and green < 20 and blue < 20

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited or not (0 <= x < width and 0 <= y < height):
            continue
        visited.add((x, y))
        if not is_background(x, y):
            continue
        pixels[x, y] = (0, 0, 0, 0)
        queue.extend(
            [
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ]
        )

    return source
