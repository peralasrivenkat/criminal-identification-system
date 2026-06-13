from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image

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
        create_page_header,
        create_panel,
        create_primary_button,
        create_secondary_button,
        render_preview,
        set_grid_weights,
    )
except ImportError:  # pragma: no cover
    from ui_theme import (
        FONTS,
        PALETTE,
        ScrollableContent,
        configure_window,
        create_page_header,
        create_panel,
        create_primary_button,
        create_secondary_button,
        render_preview,
        set_grid_weights,
    )

from database.db_operations import get_criminal, has_meaningful_criminal_details
from src.face_detection import FaceBox, draw_face_box
from src.pipeline import (
    ModelArtifactsError,
    PredictionResult,
    ensure_model_artifacts,
    identify_from_video,
    predict_frame,
    predict_image,
)
from src.preprocessing import load_image


class CriminalApp(tk.Frame):
    def __init__(self, parent: tk.Misc, *, on_back=None) -> None:
        super().__init__(parent, bg=PALETTE["bg"])
        self.parent = parent
        self.on_back = on_back
        self.image_path: Path | None = None
        self.video_path: Path | None = None
        self.live_capture = None
        self.live_running = False
        self.live_job: str | None = None
        self.live_predict_pending = False
        self.live_frame_index = 0
        self.live_processed = 0
        self.live_votes: dict[int, list[float]] = {}
        self.live_unknown_votes = 0
        self.live_source_type = "camera"
        self.live_source_ref = ""
        self._cv2 = None
        self.live_overlay_box: tuple[int, int, int, int] | None = None
        self.live_overlay_label: str | None = None
        self.video_processing = False
        self.live_artifacts = None
        self.live_tracker = None

        self._build_ui()

    def _build_ui(self) -> None:
        scrollable = ScrollableContent(self, bg=PALETTE["bg"])
        scrollable.pack(fill="both", expand=True)

        shell = tk.Frame(scrollable.content, bg=PALETTE["bg"], padx=24, pady=24)
        shell.pack(fill="both", expand=True)

        create_page_header(
            shell,
            on_back=self.on_back,
            logo_size=(600, 144),
        )

        content = tk.Frame(shell, bg=PALETTE["bg"])
        content.pack(fill="both", expand=True)
        set_grid_weights(content, columns=(0, 1), rows=(0, 1))

        control_panel = create_panel(content, padx=22, pady=22)
        control_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))

        preview_panel = create_panel(content, padx=22, pady=22)
        preview_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))

        result_panel = create_panel(content, padx=22, pady=22)
        result_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        tk.Label(
            control_panel,
            text="Detection Controls",
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            font=FONTS["title"],
        ).pack(anchor="w")
        tk.Label(
            control_panel,
            text="Upload a photo or video, or start live camera/CCTV detection from this page.",
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(8, 18))

        button_row = tk.Frame(control_panel, bg=PALETTE["card"])
        button_row.pack(fill="x")
        set_grid_weights(button_row, columns=(0, 1))
        create_primary_button(button_row, text="Upload Image", command=self.upload_image).grid(
            row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 10)
        )
        create_primary_button(button_row, text="Identify Criminal", command=self.identify).grid(
            row=0, column=1, sticky="ew", padx=(8, 0), pady=(0, 10)
        )
        create_secondary_button(button_row, text="Upload Video", command=self.upload_video).grid(
            row=1, column=0, sticky="ew", padx=(0, 8)
        )
        create_secondary_button(button_row, text="Identify From Video", command=self.identify_video).grid(
            row=1, column=1, sticky="ew", padx=(8, 0)
        )
        self.camera_button = create_secondary_button(
            button_row,
            text="Start Camera",
            command=self.start_camera,
        )
        self.camera_button.grid(row=2, column=0, sticky="ew", padx=(0, 8), pady=(10, 0))
        self.cctv_button = create_secondary_button(
            button_row,
            text="Start CCTV Stream",
            command=self.start_cctv_stream,
        )
        self.cctv_button.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.stop_live_button = create_secondary_button(
            button_row,
            text="Stop Live Detection",
            command=self.stop_live_detection,
        )
        self.stop_live_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.stop_live_button.configure(state="disabled")

        self.selection_label = tk.Label(
            control_panel,
            text="No media selected",
            bg=PALETTE["card"],
            fg=PALETTE["accent_deep"],
            font=FONTS["body_bold"],
            justify="left",
        )
        self.selection_label.pack(anchor="w", pady=(18, 0))

        tk.Label(
            preview_panel,
            text="Preview",
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            font=FONTS["title"],
        ).pack(anchor="w")
        tk.Label(
            preview_panel,
            text="Uploaded media or the live stream preview appears here.",
            bg=PALETTE["card"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(8, 16))

        self.preview_label = tk.Label(
            preview_panel,
            bg=PALETTE["surface"],
            fg=PALETTE["text_muted"],
            font=FONTS["body"],
            justify="center",
        )
        self.preview_label.pack(fill="both", expand=True)
        render_preview(self.preview_label, empty_text="Upload an image to preview it here.")

        tk.Label(
            result_panel,
            text="Prediction Result",
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            font=FONTS["title"],
        ).pack(anchor="w")
        result_body = tk.Frame(result_panel, bg=PALETTE["card"])
        result_body.pack(fill="both", expand=True, pady=(12, 0))
        self.result_text = tk.Text(
            result_body,
            bg="white",
            fg=PALETTE["text"],
            font=FONTS["body"],
            relief="flat",
            wrap="word",
            height=10,
            padx=12,
            pady=12,
        )
        result_scrollbar = ttk.Scrollbar(result_body, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        result_scrollbar.pack(side="right", fill="y")
        self._set_result_text("Results will appear here after identification.")

    def upload_image(self) -> None:
        if self.live_running:
            self._stop_live_stream(show_summary=False)
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")],
        )
        if not file_path:
            return
        self.image_path = Path(file_path)
        self.video_path = None
        render_preview(self.preview_label, path=file_path, max_size=(500, 360))
        self.selection_label.configure(text="Selected image: {0}".format(self.image_path.name))
        self._set_result_text('Image loaded. Click "Identify Criminal" to run matching.')

    def upload_video(self) -> None:
        if self.live_running:
            self._stop_live_stream(show_summary=False)
        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")],
        )
        if not file_path:
            return
        self.video_path = Path(file_path)
        self.image_path = None
        self.preview_label.configure(
            image="",
            text="Selected video\n{0}".format(self.video_path.name),
            compound="center",
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            font=FONTS["section"],
            padx=24,
            pady=84,
        )
        self.preview_label.image = None
        self.selection_label.configure(text="Selected video: {0}".format(self.video_path.name))
        self._set_result_text('Video loaded. Click "Identify From Video" to analyze it.')

    def identify(self) -> None:
        if self.live_running:
            self._stop_live_stream(show_summary=False)
        if self.image_path is None:
            messagebox.showerror("Missing Image", "Please upload an image first.")
            return

        try:
            self._set_result_text("Running prediction...")
            self.winfo_toplevel().update_idletasks()
            result = predict_image(self.image_path)
        except FileNotFoundError as error:
            messagebox.showerror("Model Missing", str(error))
            return
        except ModelArtifactsError as error:
            messagebox.showerror("Model Error", str(error))
            return
        except Exception as error:
            messagebox.showerror("Prediction Error", str(error))
            return

        self._render_image_prediction_overlay(result)
        self._show_result(result)

    def identify_video(self) -> None:
        if self.live_running:
            self._stop_live_stream(show_summary=False)
        if self.video_path is None:
            messagebox.showerror("Missing Video", "Please upload a video first.")
            return
        if self.video_processing:
            return

        self.video_processing = True
        self._set_result_text("Running video prediction...\nFaces being analyzed will be shown with a green box.")
        self.selection_label.configure(text="Analyzing video: {0}".format(self.video_path.name))
        self.winfo_toplevel().update_idletasks()
        threading.Thread(target=self._run_video_prediction, daemon=True).start()

    def _run_video_prediction(self) -> None:
        try:
            result = identify_from_video(self.video_path, on_frame=self._queue_video_preview)
        except Exception as error:
            self.after(0, self._handle_video_prediction_error, str(error))
            return
        self.after(0, self._finish_video_prediction, result)

    def _show_result(self, result: PredictionResult) -> None:
        if result.label is None:
            self._set_result_text(
                (
                    "Result: Unknown Person\n"
                    "Status: {0}\n"
                    "Confidence: {1:.2%}"
                ).format(result.status, result.confidence)
            )
            return

        criminal, images = get_criminal(result.label)
        if criminal is None:
            self._set_result_text(
                (
                    "Status: {0}\n"
                    "Predicted ID: {1}\n"
                    "Confidence: {2:.2%}\n"
                    "Record not found in database."
                ).format(result.status, result.label, result.confidence)
            )
            return

        if not has_meaningful_criminal_details(criminal):
            self._set_result_text(
                (
                    "Status: {0}\n"
                    "Predicted ID: {1}\n"
                    "Confidence: {2:.2%}\n"
                    "This ID has no full criminal profile details yet."
                ).format(result.status, criminal["id"], result.confidence)
            )
            return

        self._set_result_text(
            (
                "Status: {0}\n"
                "Predicted ID: {1}\n"
                "Confidence: {2:.2%}\n"
                "Name: {3}\n"
                "DOB: {4}\n"
                "Moles: {5}\n"
                "Nationality: {6}\n"
                "Region: {7}\n"
                "Crime: {8}\n"
                "Number of Crimes: {9}\n"
                "Database Images: {10}"
            ).format(
                result.status,
                criminal["id"],
                result.confidence,
                criminal["name"],
                criminal["dob"] or "N/A",
                criminal["moles"] or "N/A",
                criminal["nationality"] or "N/A",
                criminal["region"] or "N/A",
                criminal["crime"] or "N/A",
                criminal["num_crimes"],
                len(images),
            )
        )

    def start_camera(self) -> None:
        self._start_live_capture(
            0,
            source_type="camera",
            source_ref="camera:0",
            selection_text="Live camera running: device 0",
        )

    def start_cctv_stream(self) -> None:
        stream_source = simpledialog.askstring(
            "CCTV Stream",
            "Enter CCTV/RTSP/HTTP stream URL or camera index:",
            parent=self.winfo_toplevel(),
        )
        if not stream_source:
            return
        stream_source = stream_source.strip()
        capture_source: int | str
        if stream_source.isdigit():
            capture_source = int(stream_source)
            source_type = "camera"
            display_ref = f"camera:{capture_source}"
        else:
            capture_source = stream_source
            source_type = "cctv"
            display_ref = stream_source

        self._start_live_capture(
            capture_source,
            source_type=source_type,
            source_ref=display_ref,
            selection_text="Live stream running: {0}".format(display_ref),
        )

    def stop_live_detection(self) -> None:
        if not self.live_running:
            return
        self._stop_live_stream(show_summary=True, headline="Live detection stopped.")

    def _start_live_capture(
        self,
        capture_source: int | str,
        *,
        source_type: str,
        source_ref: str,
        selection_text: str,
    ) -> None:
        try:
            import cv2
        except ModuleNotFoundError as error:
            messagebox.showerror("OpenCV Missing", f"OpenCV is required for live detection.\n\n{error}")
            return

        self._stop_live_stream(show_summary=False)
        capture = self._open_live_capture(cv2, capture_source)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Live Source Error", f"Unable to open live source:\n{source_ref}")
            return

        self._cv2 = cv2
        try:
            self.live_artifacts = ensure_model_artifacts()
        except Exception as error:
            capture.release()
            self.live_capture = None
            self._cv2 = None
            messagebox.showerror("Model Error", str(error))
            return
        self.live_capture = capture
        self.live_running = True
        self.live_predict_pending = False
        self.live_frame_index = 0
        self.live_processed = 0
        self.live_votes = {}
        self.live_unknown_votes = 0
        self.live_source_type = source_type
        self.live_source_ref = source_ref
        self.live_overlay_box = None
        self.live_overlay_label = None
        self.live_tracker = None
        self.image_path = None
        self.video_path = None
        self.selection_label.configure(text=selection_text)
        self._set_live_button_state(running=True)
        self._set_result_text(
            "Live detection started.\n"
            f"Source: {source_ref}\n"
            "Frames will be analyzed continuously. Click \"Stop Live Detection\" to end the session."
        )
        self._poll_live_stream()

    def _open_live_capture(self, cv2, capture_source: int | str):
        if isinstance(capture_source, int) and hasattr(cv2, "CAP_DSHOW"):
            capture = cv2.VideoCapture(capture_source, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(capture_source)

    def _poll_live_stream(self) -> None:
        if not self.live_running or self.live_capture is None or self._cv2 is None:
            return

        success, frame = self.live_capture.read()
        if not success:
            self._stop_live_stream(show_summary=True, headline="Live stream ended.")
            return

        self.live_frame_index += 1
        tracked_box = self._update_live_tracker(frame)
        if tracked_box is not None:
            self.live_overlay_box = tracked_box
        elif self.live_tracker is None:
            self.live_overlay_box = None
            self.live_overlay_label = None
        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        render_preview(
            self.preview_label,
            image=Image.fromarray(self._annotated_frame(rgb_frame, self.live_overlay_box, self.live_overlay_label)),
            max_size=(500, 360),
        )

        should_predict = (
            not self.live_predict_pending
            and (
                self.live_frame_index == 1
                or self.live_frame_index % 5 == 0
                or self.live_overlay_box is None
            )
        )
        if should_predict:
            self.live_predict_pending = True
            threading.Thread(
                target=self._predict_live_frame,
                args=(rgb_frame.copy(), frame.copy(), self.live_source_type, self.live_source_ref),
                daemon=True,
            ).start()

        self.live_job = self.after(40, self._poll_live_stream)

    def _predict_live_frame(self, rgb_frame, bgr_frame, source_type: str, source_ref: str) -> None:
        try:
            prediction = predict_frame(
                rgb_frame,
                source_type=source_type,
                source_ref=source_ref,
                artifacts=self.live_artifacts,
            )
        except Exception as error:
            self.after(0, self._handle_live_prediction_error, str(error))
            return
        self.after(0, self._apply_live_prediction, prediction, rgb_frame, bgr_frame)

    def _apply_live_prediction(self, prediction: PredictionResult, rgb_frame, bgr_frame) -> None:
        self.live_predict_pending = False
        if not self.live_running:
            return

        self.live_overlay_box = prediction.face_box
        self.live_overlay_label = self._prediction_box_label(prediction)
        if prediction.face_box is not None:
            self._initialize_live_tracker(bgr_frame, prediction.face_box)
        else:
            self.live_tracker = None
        render_preview(
            self.preview_label,
            image=Image.fromarray(self._annotated_frame(rgb_frame, self.live_overlay_box, self.live_overlay_label)),
            max_size=(500, 360),
        )
        self.live_processed += 1
        if prediction.label is None:
            self.live_unknown_votes += 1
        else:
            self.live_votes.setdefault(prediction.label, []).append(prediction.confidence)

        self._set_result_text(self._build_live_status_text(current_prediction=prediction))

    def _handle_live_prediction_error(self, error_text: str) -> None:
        self.live_predict_pending = False
        if not self.live_running:
            return
        self._stop_live_stream(show_summary=False)
        messagebox.showerror("Live Detection Error", error_text)

    def _build_live_status_text(self, *, current_prediction: PredictionResult | None = None, headline: str | None = None) -> str:
        lines: list[str] = []
        if headline:
            lines.append(headline)
        lines.append("Source: {0}".format(self.live_source_ref or "camera:0"))
        lines.append("Analyzed Frames: {0}".format(self.live_processed))

        best_label, average_confidence, matches = self._get_live_leader()
        if best_label is None or self.live_unknown_votes >= max(matches, 1):
            lines.append("Live Result: Unknown Person")
            lines.append("Matched Frames: {0}".format(matches))
            lines.append("Unknown Frames: {0}".format(self.live_unknown_votes))
        else:
            criminal, images = get_criminal(best_label)
            lines.append("Live Result: Criminal identified")
            lines.append("Predicted ID: {0}".format(best_label))
            lines.append("Average Confidence: {0:.2%}".format(average_confidence))
            lines.append("Matched Frames: {0}".format(matches))
            lines.append("Unknown Frames: {0}".format(self.live_unknown_votes))
            if criminal is not None:
                lines.append("Name: {0}".format(criminal["name"] or "N/A"))
                lines.append("Crime: {0}".format(criminal["crime"] or "N/A"))
                lines.append("Region: {0}".format(criminal["region"] or "N/A"))
                lines.append("Database Images: {0}".format(len(images)))

        if current_prediction is not None:
            lines.append("")
            lines.append("Current Frame Status: {0}".format(current_prediction.status))
            lines.append("Current Frame Confidence: {0:.2%}".format(current_prediction.confidence))

        return "\n".join(lines)

    def _get_live_leader(self) -> tuple[int | None, float, int]:
        if not self.live_votes:
            return None, 0.0, 0
        best_label, confidences = max(
            self.live_votes.items(),
            key=lambda item: (len(item[1]), float(sum(item[1]) / len(item[1]))),
        )
        return int(best_label), float(sum(confidences) / len(confidences)), len(confidences)

    def _stop_live_stream(self, *, show_summary: bool, headline: str | None = None) -> None:
        summary_text = None
        if show_summary:
            summary_text = self._build_live_status_text(headline=headline or "Live detection stopped.")

        if self.live_job is not None:
            try:
                self.after_cancel(self.live_job)
            except Exception:
                pass
            self.live_job = None

        if self.live_capture is not None:
            try:
                self.live_capture.release()
            except Exception:
                pass

        self.live_capture = None
        self.live_running = False
        self.live_predict_pending = False
        self.live_frame_index = 0
        self.live_overlay_box = None
        self.live_overlay_label = None
        self.live_tracker = None
        self.live_source_type = "camera"
        self.live_source_ref = ""
        self._cv2 = None
        self.live_artifacts = None
        self._set_live_button_state(running=False)
        if show_summary and summary_text is not None:
            self.selection_label.configure(text="No media selected")
            self._set_result_text(summary_text)
        self.live_processed = 0
        self.live_votes = {}
        self.live_unknown_votes = 0

    def _create_live_tracker(self):
        if self._cv2 is None:
            return None
        factories = [
            getattr(self._cv2, "TrackerCSRT_create", None),
            getattr(self._cv2, "TrackerKCF_create", None),
            getattr(self._cv2, "TrackerMOSSE_create", None),
        ]
        legacy = getattr(self._cv2, "legacy", None)
        if legacy is not None:
            factories.extend(
                [
                    getattr(legacy, "TrackerCSRT_create", None),
                    getattr(legacy, "TrackerKCF_create", None),
                    getattr(legacy, "TrackerMOSSE_create", None),
                ]
            )
        for factory in factories:
            if callable(factory):
                try:
                    return factory()
                except Exception:
                    continue
        return None

    def _initialize_live_tracker(self, bgr_frame, face_box: tuple[int, int, int, int]) -> None:
        tracker = self._create_live_tracker()
        if tracker is None:
            self.live_tracker = None
            return
        x, y, width, height = face_box
        try:
            ok = tracker.init(bgr_frame, (float(x), float(y), float(width), float(height)))
        except Exception:
            ok = False
        self.live_tracker = tracker if ok else None

    def _update_live_tracker(self, bgr_frame) -> tuple[int, int, int, int] | None:
        if self.live_tracker is None:
            return None
        try:
            ok, box = self.live_tracker.update(bgr_frame)
        except Exception:
            self.live_tracker = None
            return None
        if not ok:
            self.live_tracker = None
            return None
        x, y, width, height = [int(round(value)) for value in box]
        if width <= 0 or height <= 0:
            self.live_tracker = None
            return None
        return (x, y, width, height)

    def _queue_video_preview(self, rgb_frame, prediction: PredictionResult, processed: int, frame_index: int) -> None:
        self.after(0, self._apply_video_preview, rgb_frame, prediction, processed, frame_index)

    def _apply_video_preview(self, rgb_frame, prediction: PredictionResult, processed: int, frame_index: int) -> None:
        annotated = self._annotated_frame(
            rgb_frame,
            prediction.face_box,
            self._prediction_box_label(prediction),
        )
        render_preview(self.preview_label, image=Image.fromarray(annotated), max_size=(500, 360))
        self._set_result_text(
            "Running video prediction...\n"
            "Processed Frames: {0}\n"
            "Frame Index: {1}\n"
            "Current Frame Status: {2}\n"
            "Current Frame Confidence: {3:.2%}".format(
                processed,
                frame_index,
                prediction.status,
                prediction.confidence,
            )
        )

    def _finish_video_prediction(self, result) -> None:
        self.video_processing = False
        self.selection_label.configure(text="Selected video: {0}".format(self.video_path.name if self.video_path else "N/A"))
        if result.label is None:
            self._set_result_text(
                (
                    "Result: Unknown Person\n"
                    "Status: {0}\n"
                    "Frames Processed: {1}\n"
                    "Matched Frames: {2}\n"
                    "Confidence: {3:.2%}"
                ).format(result.status, result.frames_processed, result.matches, result.confidence)
            )
            return

        criminal, images = get_criminal(result.label)
        if criminal is None:
            self._set_result_text(
                (
                    "Status: {0}\n"
                    "Predicted ID: {1}\n"
                    "Frames Processed: {2}\n"
                    "Matched Frames: {3}\n"
                    "Confidence: {4:.2%}"
                ).format(
                    result.status,
                    result.label,
                    result.frames_processed,
                    result.matches,
                    result.confidence,
                )
            )
            return

        self._set_result_text(
            (
                "Status: {0}\n"
                "Predicted ID: {1}\n"
                "Name: {2}\n"
                "Crime: {3}\n"
                "Region: {4}\n"
                "Frames Processed: {5}\n"
                "Matched Frames: {6}\n"
                "Confidence: {7:.2%}\n"
                "Database Images: {8}"
            ).format(
                result.status,
                criminal["id"],
                criminal["name"],
                criminal["crime"] or "N/A",
                criminal["region"] or "N/A",
                result.frames_processed,
                result.matches,
                result.confidence,
                len(images),
            )
        )

    def _handle_video_prediction_error(self, error_text: str) -> None:
        self.video_processing = False
        self.selection_label.configure(text="Selected video: {0}".format(self.video_path.name if self.video_path else "N/A"))
        messagebox.showerror("Video Prediction Error", error_text)

    def _render_image_prediction_overlay(self, result: PredictionResult) -> None:
        if self.image_path is None:
            return
        image = load_image(self.image_path)
        annotated = self._annotated_frame(image, result.face_box, self._prediction_box_label(result))
        render_preview(self.preview_label, image=Image.fromarray(annotated), max_size=(500, 360))

    def _prediction_box_label(self, prediction: PredictionResult) -> str | None:
        if prediction.face_box is None:
            return None
        if prediction.label is None:
            return "Face"
        return "ID {0}".format(prediction.label)

    def _annotated_frame(
        self,
        rgb_frame,
        face_box: tuple[int, int, int, int] | None,
        label: str | None,
    ):
        if face_box is None:
            return rgb_frame
        x, y, width, height = face_box
        return draw_face_box(
            rgb_frame,
            FaceBox(x=x, y=y, width=width, height=height),
            label=label,
        )

    def _set_live_button_state(self, *, running: bool) -> None:
        normal_state = "disabled" if running else "normal"
        stop_state = "normal" if running else "disabled"
        self.camera_button.configure(state=normal_state)
        self.cctv_button.configure(state=normal_state)
        self.stop_live_button.configure(state=stop_state)

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def destroy(self) -> None:
        self._stop_live_stream(show_summary=False)
        super().destroy()


def launch_app(container: tk.Misc | None = None, *, on_back=None) -> CriminalApp:
    if container is None:
        root = tk.Tk()
        configure_window(root, title="Criminal Identification System", geometry="1280x860")
        frame = CriminalApp(root, on_back=on_back)
        frame.pack(fill="both", expand=True)
        root.mainloop()
        return frame

    frame = CriminalApp(container, on_back=on_back)
    frame.pack(fill="both", expand=True)
    return frame


if __name__ == "__main__":
    launch_app()
