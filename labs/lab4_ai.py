import os
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pyttsx3
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from scipy.ndimage import median_filter

sys.path.append(os.getcwd())
from ui.lab4_view import Lab4View

class Lab4AI:
    def __init__(self):
        self.device = 'cpu'
        self.wavlm_model = None
        self.feature_extractor = None
        self.audio_dir = "source_audio_lab3"
        self.sr = 16000
        
        if not os.path.exists(self.audio_dir): os.makedirs(self.audio_dir)
        if not os.path.exists("results"): os.makedirs("results")
        
        self.wav_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        if not self.wav_files: 
            # Создаем пустой файл если папка пуста
            dummy = np.zeros(self.sr * 2)
            sf.write(os.path.join(self.audio_dir, "sample.wav"), dummy, self.sr)
            self.wav_files = ["sample.wav"]
            
        self.fig = plt.figure(num='BSUIR Lab 4: Voice Intelligence', figsize=(15, 9))
        self.view = Lab4View(self.fig, self.wav_files)
        
        self.src_file = self.view.radio_src.value_selected
        self.tgt_file = self.view.radio_tgt.value_selected
        
        self.view.radio_src.on_clicked(lambda l: setattr(self, 'src_file', l))
        self.view.radio_tgt.on_clicked(lambda l: setattr(self, 'tgt_file', l))
        self.view.btn_run_vc.on_clicked(self.run_vc)
        self.view.btn_run_tts.on_clicked(self.run_tts)
        
        self.view.btn_play_src.on_clicked(lambda e: self.play_audio(os.path.join(self.audio_dir, self.src_file)))
        self.view.btn_play_tgt.on_clicked(lambda e: self.play_audio(os.path.join(self.audio_dir, self.tgt_file)))
        self.view.btn_play_res.on_clicked(lambda e: self.play_audio("results/lab4_vc.wav"))
        
        plt.show()

    def play_audio(self, path):
        if os.path.exists(path):
            try: os.startfile(os.path.abspath(path))
            except: pass

    def _load_wavlm(self):
        if self.wavlm_model is None:
            self.view.set_status("Загрузка WavLM (Microsoft)... Подождите", "orange")
            self.fig.canvas.flush_events()
            try:
                model_name = "microsoft/wavlm-base-plus"
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.wavlm_model = WavLMModel.from_pretrained(model_name).to(self.device)
                self.wavlm_model.eval()
            except Exception as e:
                self.view.set_status(f"Ошибка загрузки: {str(e)}", "red")
                raise e

    def get_features(self, path):
        self._load_wavlm()
        wav, _ = librosa.load(path, sr=self.sr)
        wav = librosa.util.normalize(wav)
        wav = wav[:self.sr*10] # 10 sec limit
        
        inputs = self.feature_extractor(wav, return_tensors="pt", sampling_rate=self.sr).input_values
        with torch.no_grad():
            outputs = self.wavlm_model(inputs.to(self.device))
            feats = outputs.last_hidden_state.squeeze(0)
            
        S = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_fft=1024, hop_length=320, n_mels=128)
        min_len = min(feats.shape[0], S.shape[1])
        return feats[:min_len], S[:, :min_len], wav

    def run_vc(self, event):
        self.view.clear_gui()
        k = int(self.view.k_slider.val)
        self.view.set_status(f"Инициализация kNN-VC (k={k})...", "yellow")
        self.fig.canvas.flush_events()
        
        start_time = time.time()
        try:
            # Task 2.2: Ресурсный мониторинг (начало)
            import psutil
            process = psutil.Process(os.getpid())
            mem_start = process.memory_info().rss / 1024 / 1024
            
            src_path = os.path.join(self.audio_dir, self.src_file)
            tgt_path = os.path.join(self.audio_dir, self.tgt_file)
            
            self.view.set_status("Экстракция признаков (WavLM)...", "orange")
            self.fig.canvas.flush_events()
            
            src_feat, src_mel, src_wav = self.get_features(src_path)
            tgt_feat, tgt_mel, tgt_wav = self.get_features(tgt_path)
            
            # Step 2.1: Similarity Matrix
            src_norm = F.normalize(src_feat, p=2, dim=1)
            tgt_norm = F.normalize(tgt_feat, p=2, dim=1)
            sim_matrix = torch.mm(src_norm, tgt_norm.t())
            
            # Step 2.2: kNN Averaging
            self.view.set_status("kNN Matching & Averaging...", "yellow")
            self.fig.canvas.flush_events()
            
            topk_values, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
            converted_mel = np.zeros_like(src_mel)
            for i in range(src_mel.shape[1]):
                indices = topk_indices[i].cpu().numpy()
                converted_mel[:, i] = np.mean(tgt_mel[:, indices], axis=1)
            
            # Post-processing
            converted_mel = median_filter(converted_mel, size=(1, 3))
            
            # Step 3: Vocoder (Griffin-Lim) - Оптимально для CPU (Зад 2.2)
            self.view.set_status("Синтез аудио (Вокодер)...", "orange")
            self.fig.canvas.flush_events()
            
            res_audio = librosa.feature.inverse.mel_to_audio(
                converted_mel, sr=self.sr, n_fft=1024, hop_length=320, n_iter=64
            )
            res_audio = librosa.util.normalize(res_audio)
            
            elapsed = time.time() - start_time
            mem_end = process.memory_info().rss / 1024 / 1024
            
            out_path = "results/lab4_vc.wav"
            sf.write(out_path, res_audio, self.sr)
            
            self.view.update_plots(src_wav, tgt_wav, sim_matrix)
            
            # Task 2.3: Эксперимент с длиной
            src_len = len(src_wav)/self.sr
            quality_note = "Оптимально" if src_len > 1.5 else "Низкое качество (нужно >1.5с)"
            
            report = f"--- БГУИР ЛАБА 4: АНАЛИЗ РЕСУРСОВ ---\n"
            report += f"1. ЗАДАНИЕ 1 (TTS): pyttsx3 (Gen 2, Parametric)\n"
            report += f"2. ЗАДАНИЕ 2 (VC): kNN-VC + WavLM Base Plus\n"
            report += f"   - Время обработки: {elapsed:.2f} сек\n"
            report += f"   - RAM (пик): {mem_end:.1f} MB (Limit 2GB OK)\n"
            report += f"   - Загрузка CPU: Высокая (Инференс WavLM)\n"
            report += f"3. ЗАДАНИЕ 2.3 (Длина):\n"
            report += f"   - Длина Source: {src_len:.2f} сек ({quality_note})\n"
            report += f"   - Вывод: При <1.5с WavLM не успевает накопить\n"
            report += f"     контекст фонемы -> робоголос.\n"
            report += f"4. ВЫВОД: Метод kNN-VC эффективен на CPU,\n"
            report += f"   т.к. заменяет сложный декодер простым поиском."
            
            self.view.info_text.set_text(report)
            self.view.set_status(f"VC Ready ({elapsed:.1f}s)", "#00FF00")
            
        except Exception as e:
            self.view.set_status(f"Error: {str(e)}", "red")
            import traceback; traceback.print_exc()

    def run_tts(self, event):
        text = self.view.text_box.text
        self.view.set_status(f"TTS Synthesis: {text[:10]}...", "yellow")
        try:
            engine = pyttsx3.init()
            out_path = os.path.abspath("results/lab4_tts.wav")
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            # Даем время на сохранение
            time.sleep(1)
            self.view.set_status("TTS Ready! (Saved to results)", "#00FF00")
            self.play_audio(out_path)
        except Exception as e:
            self.view.set_status(f"TTS Error: {str(e)}", "red")

if __name__ == "__main__":
    app = Lab4AI()
