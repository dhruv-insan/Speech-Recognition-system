import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

class SpeechToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech to Text Transcription")
        self.root.geometry("500x300")
        self.audio_path = None

        self.label = tk.Label(root, text="Upload a WAV file for transcription", font=("Arial", 14))
        self.label.pack(pady=20)

        self.upload_btn = tk.Button(root, text="📁 Upload Audio", command=self.upload_audio, width=20, height=2)
        self.upload_btn.pack()

        self.transcribe_btn = tk.Button(root, text="📝 Transcribe", command=self.run_transcription, width=20, height=2)
        self.transcribe_btn.pack(pady=10)

        self.output_text = tk.Text(root, height=5, wrap='word')
        self.output_text.pack(padx=10, pady=10)

    def upload_audio(self):
        filename = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav")],
            title="Choose a .wav file"
        )
        if filename:
            self.audio_path = filename
            print("File uploaded:", self.audio_path)
            messagebox.showinfo("Uploaded", f"Selected File:\n{filename}")

    def run_transcription(self):
        if not self.audio_path:
            messagebox.showerror("Error", "Please upload a WAV file first.")
            return

        try:
            text = transcribe_audio(self.audio_path)
            if text.strip():
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"🔊 Transcription:\n{text}")
                print("🔊 Transcription:", text)
            else:
                self.output_text.insert(tk.END, "❌ No speech detected or transcription failed.")
                print("❌ Empty transcription.")
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{str(e)}")
            print("❌ Error:", e)

def transcribe_audio(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    speech, rate = sf.read(audio_path)
    if rate != 16000:
        raise ValueError("Audio must be at 16kHz sampling rate.")

    input_values = processor(speech, return_tensors="pt", sampling_rate=rate).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()
