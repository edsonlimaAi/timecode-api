from fastapi import FastAPI, UploadFile, File
import assemblyai as aai
from datetime import timedelta
import os

# Inicializa o FastAPI
app = FastAPI()

# Configuração da API key da AssemblyAI (substitua pelo seu próprio API key)
aai.settings.api_key = "805a634b1e6348e3a374213b36a74733"

# Função para converter milissegundos em timecode (HH:MM:SS:FF) com frame rate personalizável
def ms_to_timecode(ms, frame_rate=30):
    td = timedelta(milliseconds=ms)
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    frames = int((total_seconds % 1) * frame_rate)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{frames:02}"

# Função para timestamp simples no formato [HH:MM:SS]
def ms_to_simple_timestamp(ms):
    td = timedelta(milliseconds=ms)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"[{hours:02}:{minutes:02}:{seconds:02}]"

# Endpoint da API para transcrição
@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),              # Arquivo de áudio enviado
    include_srt: bool = True,                  # Incluir SRT na resposta
    include_vtt: bool = True,                  # Incluir VTT na resposta
    include_srt_text: bool = True,             # Incluir texto SRT na resposta
    include_vtt_text: bool = True,             # Incluir texto VTT na resposta
    include_simple_timestamps: bool = True,    # Incluir timestamps simples
    include_timecodes: bool = True,            # Incluir timecodes com frames
    frame_rate: float = 24.0                   # Frame rate escolhido (padrão: 24 fps)
):
    # Lista de frame rates válidos
    valid_frame_rates = [23.976, 24, 25, 30, 60]
    if frame_rate not in valid_frame_rates:
        return {"error": f"Frame rate inválido. Escolha entre {valid_frame_rates}"}

    # Salvar o arquivo temporariamente
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    # Configuração da transcrição com AssemblyAI
    config = aai.TranscriptionConfig(
        punctuate=True,           # Adicionar pontuação
        format_text=True,         # Formatar o texto
        language_detection=True,  # Detectar idioma automaticamente
        disfluencies=True         # Incluir disfluências (ex.: "uhm", "ah")
    )

    # Transcrever o áudio
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(temp_file_path)

    # Remover o arquivo temporário
    os.remove(temp_file_path)

    # Verificar se houve erro na transcrição
    if transcript.status == aai.TranscriptStatus.error:
        return {"error": transcript.error}

    # Obter sentenças da transcrição
    sentences = transcript.get_sentences()

    # Preparar a resposta
    response = {}

    # Gerar legendas SRT
    if include_srt:
        response["srt"] = transcript.export_subtitles_srt(chars_per_caption=32)

    # Gerar legendas VTT
    if include_vtt:
        response["vtt"] = transcript.export_subtitles_vtt(chars_per_caption=32)

    # Incluir texto SRT na resposta
    if include_srt_text:
        response["srt_text"] = transcript.export_subtitles_srt(chars_per_caption=32)

    # Incluir texto VTT na resposta
    if include_vtt_text:
        response["vtt_text"] = transcript.export_subtitles_vtt(chars_per_caption=32)

    # Gerar timestamps simples [HH:MM:SS]
    if include_simple_timestamps:
        simple_timestamps = [f"{ms_to_simple_timestamp(s.start)} {s.text}" for s in sentences]
        response["simple_timestamps"] = simple_timestamps

    # Gerar timecodes com frame rate escolhido (HH:MM:SS:FF)
    if include_timecodes:
        timecodes = [f"{ms_to_timecode(s.start, frame_rate)} {s.text}" for s in sentences]
        response["timecodes"] = timecodes

    # Adicionar confiança de idioma detectado
    response["language_confidence"] = transcript.json_response.get("language_confidence", "N/A")

    return response

# Para rodar localmente (opcional, descomente se quiser executar diretamente)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)