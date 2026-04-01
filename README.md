# RPG Transcriber

Uma aplicação desktop que transcreve gravações de sessões RPG, separando e organizando os áudios de jogadores e mestre.

## Funcionalidades

- 🎬 Seleção de vídeos (MKV, MP4, AVI, MOV)
- 🔊 Extração de múltiplos canais de áudio
- 📝 Transcrição com Whisper (modelo large-v3)
- 🎭 Separação de diálogos (Jogadores vs Mestre)
- 📦 Organização automática em blocos de 30 minutos
- 🖥️ Interface gráfica intuitiva com barra de progresso

## Requisitos

- Python 3.8+
- FFmpeg instalado e no PATH
- CUDA (para transcrição acelerada - opcional)
- 16GB+ de RAM (recomendado)

## Instalação

```bash
pip install faster-whisper
```

## Uso

```bash
python main.py
```

1. Clique em "Selecionar Vídeo"
2. Escolha a gravação da sessão
3. Clique em "Iniciar"
4. Os arquivos serão salvos em `output/`

## Estrutura de Saída

```
output/
├── Sessao_01/
│   ├── discord.wav
│   ├── mestre.wav
│   ├── sessao_01_parte_01.txt
│   ├── sessao_01_parte_02.txt
│   └── ...
```

## Configuração

Os canais de áudio são extraídos como:
- Track 1: Discord (Jogadores)
- Track 3: Microfone (Mestre)

Ajuste conforme necessário no código.

## Licença

MIT
