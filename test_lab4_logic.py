import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import torch.nn.functional as F
import time
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from scipy.ndimage import median_filter

def test_vc():
    device = 'cpu'
    sr = 16000
    model_name = "microsoft/wavlm-base-plus"
    
    print("Loading model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    wavlm_model = WavLMModel.from_pretrained(model_name).to(device)
    wavlm_model.eval()
    
    audio_dir = "source_audio_lab3"
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not wav_files:
        print("No wav files found")
        return
    
    src_path = os.path.join(audio_dir, wav_files[0])
    tgt_path = os.path.join(audio_dir, wav_files[min(1, len(wav_files)-1)])
    
    print(f"Processing {src_path} -> {tgt_path}")
    
    def get_features(path):
        wav, _ = librosa.load(path, sr=sr)
        wav = librosa.util.normalize(wav)
        wav = wav[:sr*5] # 5 sec for test
        
        inputs = feature_extractor(wav, return_tensors="pt", sampling_rate=sr).input_values
        with torch.no_grad():
            outputs = wavlm_model(inputs.to(device))
            feats = outputs.last_hidden_state.squeeze(0)
            
        S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=320, n_mels=128)
        min_len = min(feats.shape[0], S.shape[1])
        return feats[:min_len], S[:, :min_len], wav

    src_feat, src_mel, src_wav = get_features(src_path)
    tgt_feat, tgt_mel, tgt_wav = get_features(tgt_path)
    
    src_norm = F.normalize(src_feat, p=2, dim=1)
    tgt_norm = F.normalize(tgt_feat, p=2, dim=1)
    sim_matrix = torch.mm(src_norm, tgt_norm.t())
    
    k = 4
    topk_values, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
    converted_mel = np.zeros_like(src_mel)
    for i in range(src_mel.shape[1]):
        indices = topk_indices[i].cpu().numpy()
        converted_mel[:, i] = np.mean(tgt_mel[:, indices], axis=1)
    
    converted_mel = median_filter(converted_mel, size=(1, 3))
    
    res_audio = librosa.feature.inverse.mel_to_audio(
        converted_mel, sr=sr, n_fft=1024, hop_length=320, n_iter=32 # fewer iterations for test
    )
    
    if not os.path.exists("results"): os.makedirs("results")
    sf.write("results/test_vc.wav", res_audio, sr)
    print("Success! Result saved to results/test_vc.wav")

if __name__ == "__main__":
    test_vc()
