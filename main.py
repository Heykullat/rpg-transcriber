"""RPG Session Transcriber.

A desktop application that turns a multi-track recording of a tabletop RPG
session into a readable, timestamp-ordered transcript.

Recordings made with OBS (or similar) usually keep the remote players and the
local microphone on separate audio tracks. This tool extracts two of those
tracks with FFmpeg, transcribes each one independently with Whisper, and then
interleaves both transcripts chronologically so the conversation reads in the
order it actually happened, with every line attributed to a source.

Everything runs locally: no audio is uploaded and no API keys are required.
"""

import os
import re
import sys
import time
import wave
import threading
import traceback
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
from faster_whisper import WhisperModel

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Your own hotword list (git-ignored). Falls back to the bundled example.
HOTWORDS_FILE = os.path.join(SCRIPT_DIR, "hotwords.txt")
HOTWORDS_EXAMPLE_FILE = os.path.join(SCRIPT_DIR, "hotwords.example.txt")

SAMPLE_RATE = 16000     # Whisper expects 16 kHz mono audio
CHUNK_SECONDS = 15      # audio is fed to the model in 15 s chunks
BLOCK_MINUTES = 20      # each output file covers 20 minutes of session time

WHISPER_MODEL = "large-v3"
DEFAULT_PLAYERS_TRACK = 1
DEFAULT_GM_TRACK = 3

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
BG = "#1a1a2e"
PANEL = "#16213e"
ACCENT = "#e94560"
TEXT = "#eaeaea"
SUBTEXT = "#8892a4"
GREEN = "#4ecca3"

FONT_MAIN = ("Consolas", 10)
FONT_TITLE = ("Consolas", 13, "bold")
FONT_LOG = ("Consolas", 9)

BUTTON_STYLE = dict(font=FONT_MAIN, relief="flat", padx=14, pady=6, cursor="hand2")


# --------------------------------------------------------------------------
# Audio and text helpers (no GUI dependencies)
# --------------------------------------------------------------------------
def load_hotwords(path=None):
    """Read a hotword list and return it as a comma-separated string.

    Hotwords bias Whisper toward campaign-specific vocabulary (invented names,
    places, house rules) that a general-purpose model would otherwise mangle.

    Comment lines (``#``, ``##``, ``---``) are skipped, parenthesised notes are
    stripped so ``"Zeph (Zephyr)"`` becomes ``"Zeph"``, and duplicates are
    removed case-insensitively while preserving order.

    Args:
        path: Hotword file to read. Defaults to ``hotwords.txt``, falling back
            to ``hotwords.example.txt`` when no personal list exists.

    Returns:
        The deduplicated terms joined by ", ", or "" if no file was found.
    """
    if path is None:
        path = HOTWORDS_FILE if os.path.exists(HOTWORDS_FILE) else HOTWORDS_EXAMPLE_FILE
    if not os.path.exists(path):
        return ""

    terms, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "---", "##")):
                continue
            term = re.sub(r"\s*\(.*?\)", "", line).strip()
            if term and term.lower() not in seen:
                seen.add(term.lower())
                terms.append(term)
    return ", ".join(terms)


def load_wav(path):
    """Load a WAV file into a normalised float32 mono array.

    Whisper is fed raw samples rather than a file path so that long recordings
    can be sliced into short chunks in memory (see ``Transcriber.transcribe``).

    Args:
        path: Path to a 16-bit or 32-bit PCM WAV file.

    Returns:
        A 1-D ``numpy.ndarray`` of float32 samples in the range [-1.0, 1.0].
        Multi-channel input is downmixed to mono by averaging the channels.

    Raises:
        RuntimeError: If the sample width is neither 16-bit nor 32-bit PCM.
    """
    with wave.open(path, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes/sample")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio


def extract_audio(video_path, track, name, output_folder):
    """Extract a single audio track from a video into a 16 kHz mono WAV.

    An already-extracted file is reused, which makes re-running a session after
    a crash or a cancellation cheap.

    Args:
        video_path: Path to the source recording.
        track: Zero-based audio stream index, mapped as ``0:a:<track>``.
        name: Base name for the resulting ``<name>.wav``.
        output_folder: Directory the WAV is written to.

    Returns:
        A tuple of ``(wav_path, was_reused)``.

    Raises:
        RuntimeError: If FFmpeg exits with a non-zero status.
    """
    output = os.path.join(output_folder, f"{name}.wav")
    if os.path.exists(output):
        return output, True

    command = [
        "ffmpeg", "-i", video_path,
        "-map", f"0:a:{track}",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        output, "-y",
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed on track {track}:\n{result.stderr.decode(errors='replace')}"
        )
    return output, False


def merge_dialogue(players_blocks, gm_blocks):
    """Interleave two transcripts into a single chronological script.

    Args:
        players_blocks: Mapping of block index to ``(timestamp, line)`` tuples.
        gm_blocks: Same structure, for the game master's track.

    Returns:
        A dict mapping block index to the rendered text of that block, with
        each line prefixed by ``[Players]`` or ``[GM]`` and all lines sorted by
        their absolute timestamp within the session.
    """
    merged = {}
    for block in sorted(set(players_blocks) | set(gm_blocks)):
        lines = [(t, f"[Players] {text}") for t, text in players_blocks.get(block, [])]
        lines += [(t, f"[GM] {text}") for t, text in gm_blocks.get(block, [])]
        lines.sort(key=lambda item: item[0])
        merged[block] = "\n".join(text for _, text in lines)
    return merged


def open_folder(path):
    """Open a directory in the system file manager, creating it if needed."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    if os.name == "nt":
        os.startfile(abs_path)
    else:
        subprocess.Popen(["xdg-open", abs_path])


def windows_notify(title, message):
    """Show a Windows toast notification. No-op on other platforms."""
    if os.name != "nt":
        return

    safe_title = title.replace("'", "''")
    safe_message = message.replace("'", "''")
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager,"
        " Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;"
        "$xml = [Windows.UI.Notifications.ToastNotificationManager]"
        "::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$nodes = $xml.GetElementsByTagName('text');"
        f"$nodes.Item(0).AppendChild($xml.CreateTextNode('{safe_title}')) | Out-Null;"
        f"$nodes.Item(1).AppendChild($xml.CreateTextNode('{safe_message}')) | Out-Null;"
        "[Windows.UI.Notifications.ToastNotificationManager]"
        "::CreateToastNotifier('RPG Transcriber')"
        ".Show([Windows.UI.Notifications.ToastNotification]::new($xml))"
    )
    subprocess.Popen(
        ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class RPGTranscriberApp:
    """Tkinter front-end driving the extract → transcribe → merge pipeline.

    The pipeline runs on a worker thread so the UI stays responsive; every
    widget update from that thread is marshalled back onto the Tk event loop
    with ``root.after``. Cancellation is cooperative: ``_stop_event`` is polled
    between audio chunks, so a click on Cancel takes effect once the chunk in
    flight finishes.
    """

    def __init__(self, root):
        self.root = root
        self.video_path = None
        self._stop_event = threading.Event()
        self._log_lines = []
        self._eta = {}  # source name -> remaining-time string

        self._build_ui()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self):
        """Create every widget and wire up the button callbacks."""
        self.root.title("RPG Transcriber")
        self.root.geometry("680x640")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        tk.Label(self.root, text="⚔  RPG TRANSCRIBER  ⚔",
                 font=FONT_TITLE, bg=BG, fg=ACCENT).pack(pady=(18, 2))
        tk.Label(self.root, text="Automatic transcription of tabletop sessions",
                 font=FONT_LOG, bg=BG, fg=SUBTEXT).pack()

        self.status_var = tk.StringVar(value="Select a video to begin")
        tk.Label(self.root, textvariable=self.status_var,
                 font=FONT_MAIN, bg=BG, fg=TEXT).pack(pady=(12, 2))

        self.eta_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.eta_var,
                 font=FONT_LOG, bg=BG, fg=SUBTEXT).pack()

        bar_frame = tk.Frame(self.root, bg=BG)
        bar_frame.pack(pady=8)
        self.bar_players, self.pct_players = self._make_bar_row(bar_frame, "Players", GREEN)
        self.bar_gm, self.pct_gm = self._make_bar_row(bar_frame, "GM", ACCENT)

        track_frame = tk.Frame(self.root, bg=BG)
        track_frame.pack(pady=(0, 4))

        tk.Label(track_frame, text="Players track:", font=FONT_LOG,
                 bg=BG, fg=SUBTEXT).pack(side="left", padx=(30, 4))
        self.track_players_var = tk.IntVar(value=DEFAULT_PLAYERS_TRACK)
        tk.Spinbox(track_frame, from_=0, to=9, textvariable=self.track_players_var,
                   width=3, font=FONT_LOG, bg=PANEL, fg=TEXT,
                   buttonbackground=PANEL, relief="flat").pack(side="left", padx=(0, 20))

        tk.Label(track_frame, text="GM track:", font=FONT_LOG,
                 bg=BG, fg=SUBTEXT).pack(side="left", padx=(0, 4))
        self.track_gm_var = tk.IntVar(value=DEFAULT_GM_TRACK)
        tk.Spinbox(track_frame, from_=0, to=9, textvariable=self.track_gm_var,
                   width=3, font=FONT_LOG, bg=PANEL, fg=TEXT,
                   buttonbackground=PANEL, relief="flat").pack(side="left")

        self.output_path_var = tk.StringVar(value=f"Output: {OUTPUT_DIR}")
        tk.Label(self.root, textvariable=self.output_path_var,
                 font=FONT_LOG, bg=BG, fg=SUBTEXT).pack()

        log_frame = tk.Frame(self.root, bg=PANEL, bd=1, relief="flat")
        log_frame.pack(padx=20, pady=4, fill="both", expand=True)
        self.log_widget = tk.Text(log_frame, height=14, width=76,
                                  bg=PANEL, fg=TEXT, font=FONT_LOG,
                                  insertbackground=TEXT, relief="flat", state="disabled")
        self.log_widget.pack(padx=6, pady=6)

        self.btn_frame = tk.Frame(self.root, bg=BG)
        self.btn_frame.pack(pady=(10, 2))

        self.btn_select = tk.Button(self.btn_frame, text="📂  Select Video",
                                    bg=PANEL, fg=TEXT, activebackground=ACCENT,
                                    activeforeground="white",
                                    command=self.choose_video, **BUTTON_STYLE)
        self.btn_start = tk.Button(self.btn_frame, text="▶  Start",
                                   bg=ACCENT, fg="white", activebackground="#c73652",
                                   activeforeground="white",
                                   command=self.start, **BUTTON_STYLE)
        self.btn_cancel = tk.Button(self.btn_frame, text="✕  Cancel",
                                    bg=PANEL, fg=SUBTEXT, activebackground="#333",
                                    activeforeground=TEXT, state="disabled",
                                    command=self.cancel, **BUTTON_STYLE)
        self.btn_folder = tk.Button(self.btn_frame, text="📁  Open Folder",
                                    bg=PANEL, fg=TEXT, activebackground="#333",
                                    activeforeground=TEXT,
                                    command=lambda: open_folder(OUTPUT_DIR), **BUTTON_STYLE)
        for button in (self.btn_select, self.btn_start, self.btn_cancel, self.btn_folder):
            button.pack(side="left", padx=6)

        # Shown only once a run finishes successfully.
        self.finish_frame = tk.Frame(self.root, bg=BG)
        self.finish_frame.pack(pady=(0, 8))
        self.btn_again = tk.Button(self.finish_frame, text="🔄  Transcribe Another",
                                   bg=GREEN, fg=BG, activebackground="#3ab88e",
                                   activeforeground=BG, command=self.reset, **BUTTON_STYLE)
        self.btn_quit = tk.Button(self.finish_frame, text="⏻  Quit",
                                  bg="#333", fg=SUBTEXT, activebackground="#555",
                                  activeforeground=TEXT, command=self.quit_app, **BUTTON_STYLE)

    def _make_bar_row(self, parent, label_text, color):
        """Build one labelled progress bar row and return ``(bar, pct_label)``."""
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=30, pady=3)
        tk.Label(row, text=label_text, font=FONT_LOG, bg=BG, fg=SUBTEXT,
                 width=12, anchor="w").pack(side="left")

        style_name = f"{color}.Horizontal.TProgressbar"
        style = ttk.Style()
        style.theme_use("default")
        style.configure(style_name, troughcolor=PANEL, background=color, thickness=14)

        bar = ttk.Progressbar(row, length=460, mode="determinate", style=style_name)
        bar.pack(side="left")
        pct = tk.Label(row, text="0%", font=FONT_LOG, bg=BG, fg=SUBTEXT, width=5)
        pct.pack(side="left", padx=4)
        return bar, pct

    # -- Thread-safe UI updates -------------------------------------------
    def log(self, message):
        """Append a timestamped line to the on-screen log and the log buffer."""
        line = f"[{datetime.now():%H:%M:%S}] {message}\n"
        self._log_lines.append(line)

        def update():
            self.log_widget.configure(state="normal")
            self.log_widget.insert(tk.END, line)
            self.log_widget.see(tk.END)
            self.log_widget.configure(state="disabled")

        self.root.after(0, update)

    def set_status(self, message):
        """Set the main status line."""
        self.root.after(0, lambda: self.status_var.set(message))

    def set_eta(self, source, message):
        """Set the remaining-time estimate for one track.

        Both tracks are transcribed concurrently, so each writes into its own
        slot and the label renders whatever estimates are currently known.
        """
        if message is None:
            self._eta.pop(source, None)
        else:
            self._eta[source] = message
        text = "   |   ".join(f"{name}: {eta}" for name, eta in sorted(self._eta.items()))
        self.root.after(0, lambda: self.eta_var.set(text))

    def update_progress(self, bar, pct_label, value):
        """Move a progress bar and its percentage label to ``value`` (0-100)."""
        def update():
            bar.configure(value=value)
            pct_label.configure(text=f"{int(value)}%")

        self.root.after(0, update)

    def set_buttons(self, running):
        """Enable or disable the action buttons for the given run state."""
        def update():
            self.btn_select.configure(state="disabled" if running else "normal")
            self.btn_start.configure(state="disabled" if running else "normal")
            self.btn_cancel.configure(state="normal" if running else "disabled")

        self.root.after(0, update)

    # -- User actions ------------------------------------------------------
    def choose_video(self):
        """Prompt for the session recording to transcribe."""
        path = filedialog.askopenfilename(
            title="Select the session recording",
            filetypes=[("Videos", "*.mkv *.mp4 *.avi *.mov"), ("All files", "*.*")],
        )
        if path:
            self.video_path = path
            self.set_status(f"📹  {os.path.basename(path)}")
            self.log(f"Video selected: {path}")

    def start(self):
        """Kick off the pipeline on a background thread."""
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def cancel(self):
        """Request cancellation; takes effect after the current audio chunk."""
        self._stop_event.set()
        self.log("Cancelling... waiting for the current chunk to finish.")
        self.set_status("Cancelling...")

    def quit_app(self):
        """Stop any running work and exit."""
        self._stop_event.set()
        self.root.destroy()
        sys.exit(0)

    def on_close(self):
        """Confirm before closing while a transcription may be in progress."""
        if messagebox.askokcancel("Quit", "Close the app? Any running transcription will be cancelled."):
            self.quit_app()

    def reset(self):
        """Return the UI to its initial state so another video can be queued."""
        self.video_path = None
        self._stop_event = threading.Event()
        self._log_lines = []
        self._eta = {}

        self.btn_again.pack_forget()
        self.btn_quit.pack_forget()
        for button in (self.btn_select, self.btn_start, self.btn_cancel, self.btn_folder):
            button.pack(side="left", padx=6)

        self.status_var.set("Select a video to begin")
        self.eta_var.set("")
        self.output_path_var.set(f"Output: {OUTPUT_DIR}")
        self.update_progress(self.bar_players, self.pct_players, 0)
        self.update_progress(self.bar_gm, self.pct_gm, 0)

        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state="disabled")

    def show_finish_buttons(self):
        """Swap the action buttons for the post-run buttons."""
        for button in (self.btn_select, self.btn_start, self.btn_cancel, self.btn_folder):
            button.pack_forget()
        self.btn_again.pack(side="left", padx=10)
        self.btn_quit.pack(side="left", padx=10)

    # -- Pipeline ----------------------------------------------------------
    def load_model(self, label):
        """Load Whisper on the GPU, falling back to CPU if CUDA is unavailable.

        The CPU fallback keeps the app usable on any machine, but ``large-v3``
        in int8 mode is roughly an order of magnitude slower than float16 on a
        GPU, so the fallback is logged prominently.

        Args:
            label: Track name used in log messages.

        Returns:
            A ready-to-use ``WhisperModel``.
        """
        try:
            self.log(f"Loading Whisper {WHISPER_MODEL} (CUDA/float16) [{label}]...")
            model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            self.log(f"Model [{label}] loaded on CUDA.")
        except Exception as error:
            self.log(f"CUDA unavailable for [{label}] ({error}) — falling back to CPU/int8. "
                     f"This will be significantly slower.")
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            self.log(f"Model [{label}] loaded on CPU.")
        return model

    def transcribe(self, audio_path, model, bar, pct_label, source_name, hotwords=""):
        """Transcribe one audio track, grouped into fixed-length session blocks.

        The whole track is loaded into memory and fed to the model in
        ``CHUNK_SECONDS`` slices as raw numpy arrays. Passing short arrays
        rather than a file path avoids a CTranslate2 position-encoding issue
        that corrupts output on long inputs, and lets progress be reported
        against the real sample count.

        A failure inside a single chunk is logged and skipped rather than
        aborting the run, so one bad stretch of audio cannot cost the session.

        Args:
            audio_path: WAV file produced by :func:`extract_audio`.
            model: Whisper model instance to run.
            bar: Progress bar widget for this track.
            pct_label: Percentage label paired with ``bar``.
            source_name: Display name of the track, used in logs and the ETA.
            hotwords: Comma-separated bias terms from :func:`load_hotwords`.

        Returns:
            A dict mapping block index to a list of ``(absolute_seconds, text)``
            tuples, or ``None`` if the run was cancelled or failed outright.
        """
        try:
            self.log(f"Starting transcription: {source_name}...")
            started = time.time()

            audio = load_wav(audio_path)
            total_samples = len(audio)
            total_duration = total_samples / SAMPLE_RATE
            chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
            block_seconds = BLOCK_MINUTES * 60

            blocks = {}
            segment_count = 0
            position = 0

            while position < total_samples:
                if self._stop_event.is_set():
                    return None

                end = min(position + chunk_samples, total_samples)
                offset_seconds = position / SAMPLE_RATE

                try:
                    segments, _ = model.transcribe(
                        audio[position:end],
                        language="pt",
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 400},
                        beam_size=3,
                        word_timestamps=False,
                        condition_on_previous_text=False,
                        hotwords=hotwords or None,
                    )

                    for segment in segments:
                        text = segment.text.strip()
                        if not text:
                            continue
                        # Chunk timestamps are chunk-relative; shift to session time.
                        absolute_time = offset_seconds + segment.start
                        block_index = int(absolute_time // block_seconds)
                        blocks.setdefault(block_index, []).append((absolute_time, text))
                        segment_count += 1

                except Exception:
                    last_line = traceback.format_exc().splitlines()[-1]
                    self.log(f"  [warning] chunk at {int(offset_seconds)}s failed, skipping:\n  {last_line}")

                self.update_progress(bar, pct_label, min(int((end / total_samples) * 100), 99))

                elapsed = time.time() - started
                processed = end / SAMPLE_RATE
                speed = processed / elapsed if elapsed > 0 else 1
                self.set_eta(source_name, f"~{int((total_duration - processed) / speed)}s left")

                position = end

            self.update_progress(bar, pct_label, 100)
            self.set_eta(source_name, None)
            self.log(f"{source_name} finished in {time.time() - started:.1f}s "
                     f"({segment_count} segments).")
            return blocks

        except Exception:
            self.log(f"ERROR in {source_name}:\n{traceback.format_exc()}")
            self.set_eta(source_name, None)
            return None

    def save_log(self, session_output):
        """Write the in-memory log buffer next to the session's transcripts."""
        log_path = os.path.join(session_output, "transcription.log")
        try:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.writelines(self._log_lines)
            self.log(f"Log saved: {log_path}")
        except Exception as error:
            self.log(f"Could not save log: {error}")

    def run_pipeline(self):
        """Run the full extract → transcribe → merge → save pipeline.

        Executed on a worker thread. Both tracks are transcribed in parallel on
        two separate model instances, which roughly halves wall-clock time at
        the cost of loading the weights twice (see the VRAM note in the README).
        """
        self._stop_event = threading.Event()
        self._log_lines = []

        if not self.video_path:
            self.log("⚠  Select a video first!")
            self.set_status("No video selected")
            return

        self.set_buttons(running=True)
        self.update_progress(self.bar_players, self.pct_players, 0)
        self.update_progress(self.bar_gm, self.pct_gm, 0)

        try:
            # Name the output folder after the last number in the file name,
            # e.g. "campaign session 07.mkv" -> "Session_07".
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            numbers = re.findall(r"\d+", video_name)
            session_number = int(numbers[-1]) if numbers else 0
            session_folder = (f"Session_{session_number:02d}" if numbers
                              else video_name.replace(" ", "_"))
            session_output = os.path.join(OUTPUT_DIR, session_folder)
            os.makedirs(session_output, exist_ok=True)

            self.root.after(0, lambda: self.output_path_var.set(f"Output: {session_output}"))
            self.log(f"Output folder: {session_output}")

            # 1. Extract both audio tracks.
            self.set_status("Extracting audio...")
            tracks = [
                (self.track_players_var.get(), "players", "Players"),
                (self.track_gm_var.get(), "gm", "GM"),
            ]
            audio_paths = {}
            for track, file_name, display_name in tracks:
                self.log(f"Extracting audio: {display_name} (track {track})...")
                path, reused = extract_audio(self.video_path, track, file_name, session_output)
                self.log(f"Audio '{display_name}' {'reused from disk' if reused else 'extracted'}.")
                audio_paths[display_name] = path

            if self._stop_event.is_set():
                self.set_status("Cancelled.")
                return

            # 2. Load campaign vocabulary.
            hotwords = load_hotwords()
            if hotwords:
                self.log(f"Hotwords loaded: {len(hotwords.split(', '))} terms.")
            else:
                self.log("No hotword file found — continuing without vocabulary bias.")

            # 3. Load one model per track.
            self.set_status(f"Loading Whisper {WHISPER_MODEL}...")
            models = {name: self.load_model(name) for name in ("Players", "GM")}

            # 4. Transcribe both tracks concurrently.
            self.set_status("Transcribing both tracks...")
            results = {}
            widgets = {
                "Players": (self.bar_players, self.pct_players),
                "GM": (self.bar_gm, self.pct_gm),
            }

            def worker(name):
                bar, pct_label = widgets[name]
                results[name] = self.transcribe(
                    audio_paths[name], models[name], bar, pct_label, name, hotwords
                )

            threads = [threading.Thread(target=worker, args=(name,), daemon=True)
                       for name in ("Players", "GM")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            if self._stop_event.is_set():
                self.set_status("Cancelled.")
                self.eta_var.set("")
                return

            players_blocks = results.get("Players") or {}
            gm_blocks = results.get("GM") or {}
            if not players_blocks and not gm_blocks:
                raise RuntimeError("Both transcriptions failed or returned nothing.")

            # 5. Merge and write one file per block.
            self.set_status("Merging dialogue...")
            self.root.after(0, lambda: self.eta_var.set(""))
            merged = merge_dialogue(players_blocks, gm_blocks)

            for block, text in merged.items():
                file_path = os.path.join(
                    session_output,
                    f"session_{session_number:02d}_part_{block + 1:02d}.txt",
                )
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                self.log(f"✅ Saved: {file_path}")

            self.save_log(session_output)
            self.set_status(f"✅  Done! {len(merged)} part(s) in {session_folder}/")
            self.log("=" * 50)
            self.log("ALL DONE! Click 📁 Open Folder to see the files.")
            self.log("=" * 50)

            windows_notify("RPG Transcriber — Done!",
                           f"{len(merged)} part(s) saved to {session_folder}")
            self.root.after(0, self.show_finish_buttons)

        except Exception:
            error_text = traceback.format_exc()
            self.log(f"FATAL ERROR:\n{error_text}")
            self.set_status("❌  Error — check the log.")
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Something went wrong:\n\n{error_text}"))

        finally:
            self.set_buttons(running=False)


def main():
    """Create the output directory and start the Tk event loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    root = tk.Tk()
    RPGTranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
