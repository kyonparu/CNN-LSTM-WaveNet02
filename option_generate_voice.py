import os
import numpy as np
import pyworld as pw
import soundfile as sf
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_voice_from_saved_data(input_dir, output_dir, sampling_rate=50000):
    """
    テストデータの音声を生成して保存する関数。

    Args:
        input_dir (str): 合成用データが保存されているディレクトリ（.npzファイル）。
        output_dir (str): 生成した音声を保存するディレクトリ。
        sampling_rate (int): サンプリング周波数（デフォルトは50kHz）。
    """
    # 入力ディレクトリが存在するか確認
    if not os.path.exists(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist.")
        exit(1)

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力ディレクトリ内のすべての.npzファイルを取得
    files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    if not files:
        logging.warning(f"No .npz files found in {input_dir}.")
        return

    for file in files:
        # ファイルパスを生成
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".npz", ".wav"))

        # .npzファイルを読み込む
        data = np.load(input_path)
        f0 = data["f0"]
        spectral_envelope = data["spectral_envelope"]
        aperiodicity = data["aperiodicity"]

        # 音声を生成
        logging.info(f"Generating voice for {file}...")
        waveform = pw.synthesize(f0, spectral_envelope, aperiodicity, sampling_rate)

        # WAVファイルとして保存
        sf.write(output_path, waveform, sampling_rate)
        logging.info(f"Saved generated voice to {output_path}")

if __name__ == "__main__":
    # 入力ディレクトリと出力ディレクトリを指定
    input_dir = "inference_output"  # 合成用データが保存されているディレクトリ
    output_dir = "generated_wav"   # 生成した音声を保存するディレクトリ

    # 音声を生成して保存
    generate_voice_from_saved_data(input_dir, output_dir)

