import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from faster_whisper import WhisperModel

VIDEO = None
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- GUI ----------
root = tk.Tk()
root.title("Transcrição RPG")
root.geometry("600x450")

progress_discord = ttk.Progressbar(root, length=500, mode="determinate")
progress_mestre = ttk.Progressbar(root, length=500, mode="determinate")

label = tk.Label(root, text="Selecione um vídeo para começar")
log = tk.Text(root, height=18, width=70)

label.pack(pady=5)
progress_discord.pack(pady=5)
progress_mestre.pack(pady=5)
log.pack(pady=10)

def log_msg(msg):
    def _update():
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
    root.after(0, _update)

def set_label(msg):
    root.after(0, lambda: label.config(text=msg))

def update_progress(bar, value):
    root.after(0, lambda: bar.configure(value=value))

# ---------- SELECIONAR VÍDEO ----------
def choose_video():
    global VIDEO

    arquivo = filedialog.askopenfilename(
        title="Selecione a gravação da sessão",
        filetypes=[
            ("Vídeos", "*.mkv *.mp4 *.avi *.mov"),
            ("Todos os arquivos", "*.*")
        ]
    )

    if arquivo:
        VIDEO = arquivo
        nome = os.path.basename(arquivo)
        set_label(f"Arquivo selecionado: {nome}")
        log_msg(f"Vídeo escolhido: {arquivo}")

# ---------- FUNÇÕES ----------
def extract_audio(track, name, output_folder):
    output = os.path.join(output_folder, f"{name}.wav")

    if os.path.exists(output):
        log_msg(f"{name} já existe")
        return output

    log_msg(f"Extraindo {name}...")

    cmd = [
        "ffmpeg",
        "-i", VIDEO,
        "-map", f"0:a:{track}",
        "-ar", "16000",
        "-ac", "1",
        output,
        "-y"
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output

def transcribe(audio_path, model, progress_bar):
    segments, _ = model.transcribe(
        audio_path,
        language="pt",
        vad_filter=True,
        beam_size=5
    )

    segments = list(segments)
    total = len(segments)

    blocos = {}
    duracao_bloco = 30 * 60

    for i, seg in enumerate(segments, 1):
        linha = seg.text.strip()
        bloco_idx = int(seg.start // duracao_bloco)

        if bloco_idx not in blocos:
            blocos[bloco_idx] = []

        blocos[bloco_idx].append((seg.start, linha))

        progress = int((i / total) * 100)
        update_progress(progress_bar, progress)

    update_progress(progress_bar, 100)
    return blocos

def merge_dialogues(discord_blocos, mic_blocos):
    blocos_finais = {}

    todos_blocos = set(discord_blocos.keys()) | set(mic_blocos.keys())

    for bloco in sorted(todos_blocos):
        linhas = []

        for tempo, fala in discord_blocos.get(bloco, []):
            linhas.append((tempo, f"[Jogadores] {fala}"))

        for tempo, fala in mic_blocos.get(bloco, []):
            linhas.append((tempo, f"[Mestre] {fala}"))

        linhas.sort(key=lambda x: x[0])
        texto = "\n".join(fala for _, fala in linhas)

        blocos_finais[bloco] = texto

    return blocos_finais

# ---------- MAIN ----------
def run():
    if not VIDEO:
        log_msg("Selecione um vídeo primeiro!")
        set_label("Nenhum vídeo selecionado")
        return

    nome_video = os.path.splitext(os.path.basename(VIDEO))[0]

    match = re.search(r"(\d+)", nome_video)

    if match:
        numero_sessao = int(match.group(1))
        pasta_sessao = f"Sessao_{numero_sessao:02d}"
    else:
        numero_sessao = 0
        pasta_sessao = nome_video.replace(" ", "_")

    session_output = os.path.join(OUTPUT_DIR, pasta_sessao)
    os.makedirs(session_output, exist_ok=True)

    set_label("Extraindo áudios...")

    discord_audio = extract_audio(1, "discord", session_output)
    mic_audio = extract_audio(3, "mestre", session_output)

    set_label("Carregando modelo...")
    log_msg("Carregando Whisper large-v3...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    set_label("Transcrevendo Discord...")
    discord_blocos = transcribe(
        discord_audio,
        model,
        progress_discord
    )

    set_label("Transcrevendo Mestre...")
    mic_blocos = transcribe(
        mic_audio,
        model,
        progress_mestre
    )

    set_label("Juntando diálogos...")
    merged_blocos = merge_dialogues(
        discord_blocos,
        mic_blocos
    )

    for bloco, texto in merged_blocos.items():
        nome_arquivo = os.path.join(
            session_output,
            f"sessao_{numero_sessao:02d}_parte_{bloco + 1:02d}.txt"
        )

        with open(nome_arquivo, "w", encoding="utf-8") as f:
            f.write(texto)

        log_msg(f"Arquivo salvo: {nome_arquivo}")

    set_label("Finalizado!")
    log_msg("Tudo pronto!")

# ---------- BOTÕES ----------
btn_select = tk.Button(
    root,
    text="Selecionar Vídeo",
    command=choose_video
)

btn_start = tk.Button(
    root,
    text="Iniciar",
    command=lambda: threading.Thread(
        target=run,
        daemon=True
    ).start()
)

btn_select.pack(pady=5)
btn_start.pack(pady=5)

root.mainloop()