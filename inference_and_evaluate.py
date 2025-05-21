import torch
import soundfile as sf
import pyworld as pw
import numpy as np
from dataset import SpeechDataset
from model import CNN_LSTM_WaveNet
from scipy.fftpack import dct
import os
import yaml
from utils import extract_speaker_and_sentence_id  # 追加
import logging
import matplotlib.pyplot as plt

# ログの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# モデルとデータセットの準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, device):
    model = CNN_LSTM_WaveNet(
        in_channels=6,  # チェックポイントと一致させる
        cnn_channels=64,
        lstm_hidden=128,
        output_dim=261,  # チェックポイントと一致させる
        wavenet_channels=128,
        embed_dim=16  # チェックポイントと一致させる
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_audio(model, dataset, speaker_id, sentence_id, device):
    # データセットから特徴量を取得
    combined_audio_features, articulation_features, phoneme_embeddings = dataset(speaker_id, sentence_id)
    
    # CNN用の調音特徴量
    articulation_features = torch.stack([articulation_features]).float()  # [1, time_steps, 6, 6]
    articulation_features = articulation_features.permute(0, 3, 1, 2).to(device)  # [1, 6, 6, time_steps]
    
    # 言語特徴量
    phoneme_embeddings = torch.stack([phoneme_embeddings]).float().to(device)  # [1, time_steps, embed_dim]
    
    # デバッグ用ログ
    logging.info(f"articulation_features shape: {articulation_features.shape}")
    logging.info(f"phoneme_embeddings shape: {phoneme_embeddings.shape}")
    
    with torch.no_grad():
        # モデルの推論
        outputs = model(articulation_features, phoneme_embeddings)
        logging.info(f"Model outputs type: {type(outputs)}")
        logging.info(f"Model outputs shape: {outputs.shape}")
        
        # テンソル型の処理
        f0 = outputs[0, :, 0].detach().cpu().numpy()  # F0は1次元目
        spectral_envelope = outputs[0, :, 1:60].detach().cpu().numpy()  # スペクトル包絡
        aperiodicity = outputs[0, :, 60:].detach().cpu().numpy()  # 非周期性成分

        # デバッグログ: 生成直後
        logging.info(f"Initial f0 shape: {f0.shape}, dtype: {f0.dtype}, C-contiguous: {f0.flags['C_CONTIGUOUS']}")
        logging.info(f"Initial spectral_envelope shape: {spectral_envelope.shape}, dtype: {spectral_envelope.dtype}, C-contiguous: {spectral_envelope.flags['C_CONTIGUOUS']}")
        logging.info(f"Initial aperiodicity shape: {aperiodicity.shape}, dtype: {aperiodicity.dtype}, C-contiguous: {aperiodicity.flags['C_CONTIGUOUS']}")

        # 次元の調整
        if spectral_envelope.shape[1] != aperiodicity.shape[1]:
            min_dim = min(spectral_envelope.shape[1], aperiodicity.shape[1])
            spectral_envelope = spectral_envelope[:, :min_dim]
            aperiodicity = aperiodicity[:, :min_dim]

        # C連続性とデータ型を保証
        spectral_envelope = np.ascontiguousarray(spectral_envelope.copy().astype(np.float64))
        aperiodicity = np.ascontiguousarray(aperiodicity.copy().astype(np.float64))

        # デバッグログ
        logging.info(f"Adjusted spectral_envelope shape: {spectral_envelope.shape}, dtype: {spectral_envelope.dtype}, C-contiguous: {spectral_envelope.flags['C_CONTIGUOUS']}")
        logging.info(f"Adjusted aperiodicity shape: {aperiodicity.shape}, dtype: {aperiodicity.dtype}, C-contiguous: {aperiodicity.flags['C_CONTIGUOUS']}")

        # MCEP → SP に変換
        fft_size = 1024  # FFTサイズ
        spectral_envelope = pw.decode_spectral_envelope(
            spectral_envelope, 50000, fft_size
        )
        aperiodicity = pw.decode_spectral_envelope(
            aperiodicity, 50000, fft_size
        )

        # デバッグログ: decode_spectral_envelope後
        logging.info(f"Decoded spectral_envelope shape: {spectral_envelope.shape}, dtype: {spectral_envelope.dtype}, C-contiguous: {spectral_envelope.flags['C_CONTIGUOUS']}")
        logging.info(f"Decoded aperiodicity shape: {aperiodicity.shape}, dtype: {aperiodicity.dtype}, C-contiguous: {aperiodicity.flags['C_CONTIGUOUS']}")

        # 値の検証と修正
        f0 = np.maximum(f0, 0)
        spectral_envelope = np.maximum(spectral_envelope, 1e-10)
        aperiodicity = np.maximum(aperiodicity, 1e-10)

        # データ型をnp.float64に変換し、C連続配列に変換
        f0 = np.ascontiguousarray(f0.copy().astype(np.float64))
        spectral_envelope = np.ascontiguousarray(spectral_envelope.copy().astype(np.float64))
        aperiodicity = np.ascontiguousarray(aperiodicity.copy().astype(np.float64))

        # デバッグログ
        logging.info(f"f0 is C-contiguous: {f0.flags['C_CONTIGUOUS']}")
        logging.info(f"spectral_envelope is C-contiguous: {spectral_envelope.flags['C_CONTIGUOUS']}")
        logging.info(f"aperiodicity is C-contiguous: {aperiodicity.flags['C_CONTIGUOUS']}")

        # 音声の出力
        output_waveform = pw.synthesize(f0, spectral_envelope, aperiodicity, 50000)
    return output_waveform, f0, spectral_envelope, aperiodicity

def load_testlist(testlist_path):
    with open(testlist_path, 'r') as f:
        lines = f.readlines()
    test_items = [line.strip() for line in lines]
    return test_items

def calculate_mcd(target, generated):
    """Calculate Mel-Cepstral Distortion (MCD)"""
    mcd = 0
    for t, g in zip(target, generated):
        mcd += np.sqrt(np.sum((t - g)**2))
    return (10.0 / np.log(10)) * (mcd / len(target))

def calculate_gpe(target_f0, generated_f0, threshold=0.2):
    """Calculate Gross Pitch Error (GPE)"""
    target_voiced = target_f0 > 0
    generated_voiced = generated_f0 > 0
    errors = np.abs(target_f0 - generated_f0) / (target_f0 + 1e-10)
    gpe = np.sum((errors > threshold) & target_voiced & generated_voiced) / np.sum(target_voiced)
    return gpe

def calculate_vde(target_voiced, generated_voiced):
    """Calculate Voicing Decision Error (VDE)"""
    return np.sum(target_voiced != generated_voiced) / len(target_voiced)

def calculate_ffe(target_f0, generated_f0):
    """Calculate F0 Frame Error (FFE)"""
    return np.sum((target_f0 == 0) != (generated_f0 == 0)) / len(target_f0)

def plot_mel_spectrogram(target_spectrogram, predicted_spectrogram, save_path, speaker_id, sentence_id):
    """
    メルスペクトログラムをプロットして保存する関数
    :param target_spectrogram: ターゲットのスペクトログラム
    :param predicted_spectrogram: 推定されたスペクトログラム
    :param save_path: 保存先のパス
    :param speaker_id: スピーカーID
    :param sentence_id: 発話ID
    """
    plt.figure(figsize=(12, 6))

    # ターゲットのスペクトログラム
    plt.subplot(1, 2, 1)
    plt.imshow(target_spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.title(f"Target Spectrogram\nSpeaker: {speaker_id}, Sentence: {sentence_id}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    # 推定されたスペクトログラム
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.title(f"Predicted Spectrogram\nSpeaker: {speaker_id}, Sentence: {sentence_id}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    # 保存
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Spectrogram plot saved to {save_path}")

def plot_mel_cepstrum(mel_cepstrum, save_path, speaker_id, sentence_id):
    """
    メルケプストラムをプロットして保存する関数
    :param mel_cepstrum: メルケプストラム（2次元配列）
    :param save_path: 保存先のパス
    :param speaker_id: スピーカーID
    :param sentence_id: 発話ID
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_cepstrum.T, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    plt.colorbar(label="Amplitude")
    plt.title(f"Mel-Cepstrum\nSpeaker: {speaker_id}, Sentence: {sentence_id}")
    plt.xlabel("Time")
    plt.ylabel("Mel-Cepstrum Coefficients")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Mel-Cepstrum plot saved to {save_path}")

def evaluate_generated_audio(config):
    logging.info("Starting evaluation process...")
    
    # モデルのロード
    logging.info("Loading model...")
    model = load_model(config['checkpoint'], device)
    logging.info("Model loaded successfully.")

    # データセットの準備
    logging.info("Loading dataset...")
    dataset = SpeechDataset(data_dir=config['data_dir'])
    logging.info("Dataset loaded successfully.")

    # テストリストの読み込み
    logging.info("Loading test list...")
    test_items = load_testlist(config['testlist'])
    logging.info(f"Test list loaded successfully. {len(test_items)} items found.")

    # inference_outputディレクトリを作成
    inference_output_dir = config.get('inference_output_dir', 'inference_output')
    os.makedirs(inference_output_dir, exist_ok=True)

    # メルケプストラムの保存先ディレクトリ
    mel_cepstrum_dir = os.path.join(inference_output_dir, "mel_cepstrum")
    os.makedirs(mel_cepstrum_dir, exist_ok=True)

    # 評価指標の初期化
    mcd_scores = []
    gpe_scores = []
    vde_scores = []
    ffe_scores = []

    for idx, item in enumerate(test_items):
        logging.info(f"Processing item {idx + 1}/{len(test_items)}: {item}")

        # ファイル名からスピーカーIDと発話番号を抽出
        speaker_id, sentence_id = extract_speaker_and_sentence_id(item)
        if speaker_id is None or sentence_id is None:
            logging.warning(f"Skipping item {item}: Unable to extract speaker ID or sentence ID.")
            continue

        # ターゲット音声のパスを生成
        target_audio_path = os.path.join(
            config['data_dir'], "audio_data", speaker_id, f"ATR503{speaker_id}_{sentence_id}.wav"
        )
        if not os.path.exists(target_audio_path):
            logging.warning(f"Target audio file not found: {target_audio_path}")
            continue

        # ターゲット音声を読み込む
        logging.info(f"Loading target audio: {target_audio_path}")
        target_waveform, _ = sf.read(target_audio_path)
        logging.info("Target audio loaded successfully.")

        # 推論結果を生成
        logging.info(f"Generating audio for speaker {speaker_id}, sentence {sentence_id}...")
        generated_waveform, f0, mgc, aperiodicity = generate_audio(model, dataset, speaker_id, sentence_id, device)
        logging.info("Audio generation completed.")

        # メルケプストラムを保存
        mel_cepstrum_path = os.path.join(mel_cepstrum_dir, f"{speaker_id}_{sentence_id}_mel_cepstrum.npy")
        np.save(mel_cepstrum_path, mgc)
        logging.info(f"Mel-Cepstrum saved: {mel_cepstrum_path}")

        # メルケプストラムをプロットして保存
        mel_cepstrum_plot_path = os.path.join(mel_cepstrum_dir, f"{speaker_id}_{sentence_id}_mel_cepstrum_plot.png")
        plot_mel_cepstrum(mgc, mel_cepstrum_plot_path, speaker_id, sentence_id)
        logging.info(f"Mel-Cepstrum plot saved: {mel_cepstrum_plot_path}")

        # デバッグログ
        logging.info(f"Target waveform length: {len(target_waveform)}")
        logging.info(f"Generated waveform length: {len(generated_waveform)}")

        # 長さを揃える（ゼロパディング）
        max_length = max(len(target_waveform), len(generated_waveform))
        target_waveform = np.pad(target_waveform, (0, max_length - len(target_waveform)), mode='constant')
        generated_waveform = np.pad(generated_waveform, (0, max_length - len(generated_waveform)), mode='constant')

        # デバッグログ: パディング後の長さ
        logging.info(f"Padded target waveform length: {len(target_waveform)}")
        logging.info(f"Padded generated waveform length: {len(generated_waveform)}")

        # 評価指標を計算
        logging.info("Calculating evaluation metrics...")
        mcd = calculate_mcd(target_waveform, generated_waveform)
        mcd_scores.append(mcd)
        logging.info(f"MCD calculated: {mcd:.4f}")

        # F0（基本周波数）を抽出
        target_f0, _ = pw.dio(target_waveform, fs=50000)
        generated_f0, _ = pw.dio(generated_waveform, fs=50000)

        # Voicing情報を計算
        target_voiced = target_f0 > 0
        generated_voiced = generated_f0 > 0

        # GPE, VDE, FFEを計算
        gpe = calculate_gpe(target_f0, generated_f0)
        gpe_scores.append(gpe)
        logging.info(f"GPE calculated: {gpe:.4f}")

        vde = calculate_vde(target_voiced, generated_voiced)
        vde_scores.append(vde)
        logging.info(f"VDE calculated: {vde:.4f}")

        ffe = calculate_ffe(target_f0, generated_f0)
        ffe_scores.append(ffe)
        logging.info(f"FFE calculated: {ffe:.4f}")

        # 必要な情報を保存
        output_path = os.path.join(inference_output_dir, f"{speaker_id}_{sentence_id}.npz")
        np.savez(output_path, f0=f0, mgc=mgc, aperiodicity=aperiodicity)
        logging.info(f"Saved synthesis data for {speaker_id}_{sentence_id} to {output_path}")

    # 評価結果をログに出力（要約情報）
    logging.info("MCD scores: mean={:.4f}, min={:.4f}, max={:.4f}".format(
    np.mean(mcd_scores), min(mcd_scores), max(mcd_scores)))
    logging.info("GPE scores: mean={:.4f}, min={:.4f}, max={:.4f}".format(
    np.mean(gpe_scores), min(gpe_scores), max(gpe_scores)))
    logging.info("VDE scores: mean={:.4f}, min={:.4f}, max={:.4f}".format(
    np.mean(vde_scores), min(vde_scores), max(vde_scores)))
    logging.info("FFE scores: mean={:.4f}, min={:.4f}, max={:.4f}".format(
    np.mean(ffe_scores), min(ffe_scores), max(ffe_scores)))

    # 評価結果をログに出力（詳細情報）
    #logging.debug(f"MCD scores (detailed): {mcd_scores}")
    #logging.debug(f"GPE scores (detailed): {gpe_scores}")
    #logging.debug(f"VDE scores (detailed): {vde_scores}")
    #logging.debug(f"FFE scores (detailed): {ffe_scores}")

    # 評価結果を返す
    logging.info("Evaluation process completed.")
    return mcd_scores, gpe_scores, vde_scores, ffe_scores

if __name__ == "__main__":
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logging.info("Starting evaluation script...")
    metrics = evaluate_generated_audio(config)

    # numpy.float64をfloatに変換
    metrics = tuple([float(score) for score in metric_list] for metric_list in metrics)

    # ログに出力
    #logging.info(f"MCD scores: [{', '.join([f'{score:.4f}' for score in metrics[0]])}]")
    #logging.info(f"GPE scores: [{', '.join([f'{score:.4f}' for score in metrics[1]])}]")
    #logging.info(f"VDE scores: [{', '.join([f'{score:.4f}' for score in metrics[2]])}]")
    #logging.info(f"FFE scores: [{', '.join([f'{score:.4f}' for score in metrics[3]])}]")
    #logging.info("Evaluation script completed.")
