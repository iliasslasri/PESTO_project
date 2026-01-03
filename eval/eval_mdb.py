from mirdataset import MIRDataset
from pesto import load_model
import torch
import mir_eval
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    fs = 44100
    hpsz = 10 # ms
    pesto_model = load_model("mir-1k_g7", step_size=hpsz, sampling_rate=fs)
    pesto_model.eval()
    mix_levels = [0.0, 0.1, 0.2, 0.3, 0.6, 0.9]

    metrics = {}
    for mix in mix_levels:
        count = 0
        total_rpa = 0
        total_rca = 0
        total_oa = 0
        metrics[mix] = {}
        metrics[mix]["RPA"] = 0
        metrics[mix]["RCA"] = 0
        metrics[mix]["OA"] = 0
        print(f"\n--- Evaluating Reverb Mix: {mix} ---")
        md = torch.utils.data.DataLoader(MIRDataset("mdb_stem_synth", mix=mix), batch_size=1, shuffle=True, drop_last=False)
        for audio, times, freqs in tqdm(md):
            with torch.no_grad():
                f0_pred, conf, amp = pesto_model(
                audio,
                convert_to_freq=True,
                return_activations=False,
            )
            f0_pred = np.nan_to_num(f0_pred, nan=0.0)

            times_pred = np.arange(f0_pred.shape[-1]) * (hpsz / 1000.0)
            times_pred = times_pred.flatten()
            f0_pred = f0_pred.flatten()
            times = times.numpy().flatten()
            freqs = freqs.numpy().flatten()
            scores = mir_eval.melody.evaluate(times, freqs, times_pred, f0_pred)
            total_rpa += scores['Raw Pitch Accuracy']
            total_rca += scores['Raw Chroma Accuracy']
            total_oa += scores['Overall Accuracy']
            count += 1
        if count > 0:
            avg_rpa = total_rpa / count
            avg_rca = total_rca / count
            avg_oa = total_oa / count
            print(f"Result (Mix {mix}): RPA={avg_rpa*100:.2f}% | RCA={avg_rca*100:.2f}%")
            metrics[mix]["RPA"] = avg_rpa*100
            metrics[mix]["RCA"] = avg_rca*100
            metrics[mix]["OA"] = avg_oa*100
        else:
            print("No valid samples evaluated.")

    df = pd.DataFrame.from_dict(metrics, orient='index')
    df.index.name = 'Mix Level'
    
    df.to_csv('results.csv')



if __name__ == "__main__":
    main()