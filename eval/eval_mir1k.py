import torch
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
from pesto import load_model
import mir_eval
from mir1k_dataset import MIR1KDataset

HOP_SIZE_SECONDS = 0.020
pesto_model = load_model(
    'mir-1k_g7',
    step_size=20.,
    sampling_rate=16000, # mir-1k is in sampled @16k
    max_batch_size=4
)


def run_evaluation(dataset_root):
    
    mix_levels = [0.0, 0.1, 0.2, 0.3, 0.6, 0.9]
    
    for mix in mix_levels:
        print(f"\n--- Evaluating Reverb Mix: {mix} ---")

        dataset = MIR1KDataset(dataset_root, mix=mix, target_sr=16000)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Accumulators for metrics
        total_rpa = 0.0
        total_rca = 0.0
        count = 0
        metrics = {}
        metrics["RPA"] = 0
        metrics["RCA"] = 0
        metrics["OA"] = 0
        for batch in tqdm(loader):
            x, y = batch
            # x: [1, 1, T], y: [1, L]
            
            audio_input = x.squeeze(0) # [1, T]
            if count==0:
                sf.write(f"audio_{mix}.wav", audio_input[0].detach().cpu().numpy(),  samplerate=16000)

            with torch.no_grad():
                # Get Pitch (Hz) and Confidence
                pitch, conf, _ = pesto_model(audio_input, convert_to_freq=True, return_activations=False)
            import ipdb
            # --- Post-Processing ---
            pred_hz = pitch.cpu().numpy().flatten()
            conf_np = conf.cpu().numpy().flatten()
            ref_hz = y.numpy().flatten()

            # Apply Voicing Threshold
            # Set unconfident predictions to 0 (Unvoiced)
            est_freq = pred_hz.copy()
            # est_freq[conf_np < 0.2] = 0.0
            
            # Generate Timestamps
            # We construct time arrays so mir_eval knows exactly where each frame sits.
            est_time = np.arange(len(est_freq)) * HOP_SIZE_SECONDS
            ref_time = np.arange(len(ref_hz)) * HOP_SIZE_SECONDS

            # mir_eval handles the interpolation/alignment automatically based on the timestamps
            # We skip files where ground truth is empty or invalid
            if len(ref_hz) > 0 and np.sum(ref_hz) > 0:
                scores = mir_eval.melody.evaluate(ref_time, ref_hz, est_time, est_freq)
                total_rpa += scores['Raw Pitch Accuracy']
                total_rca += scores['Raw Chroma Accuracy']
                count += 1

        if count > 0:
            avg_rpa = total_rpa / count
            avg_rca = total_rca / count
            print(f"Result (Mix {mix}): RPA={avg_rpa*100:.2f}% | RCA={avg_rca*100:.2f}%")
        else:
            print("No valid samples evaluated.")

if __name__ == "__main__":
    MIR_1K_PATH = "./MIR-1K"
    
    if os.path.exists(MIR_1K_PATH):
        run_evaluation(MIR_1K_PATH)
    else:
        print("Please configure dataset path in the script.\n It can be downloaded from http://mirlab.org/dataset/public/ .")