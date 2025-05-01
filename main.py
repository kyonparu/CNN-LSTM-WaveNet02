import os
import argparse
import logging
import yaml
from train import train_model
from inference_and_evaluate import evaluate_generated_audio, load_testlist
from dataset import extract_features, SpeechDataset
from model import CNN_LSTM_WaveNet
from utils import load_checkpoint, print_metrics
import torch
from datetime import datetime

# ログ設定
log_dir = "log_history"
os.makedirs(log_dir, exist_ok=True)  # ログディレクトリを作成

# 現在時刻を取得してログファイル名を設定
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"pipeline_{current_time}.log")

# ロガーを取得
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # ログレベルを INFO に設定

# フォーマットを定義
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# ファイルハンドラーを設定
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ストリームハンドラー（ターミナル出力）を設定
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 設定ファイルの読み込み
    config = load_config(args.config)
    logging.info("Configuration loaded.")

    data_dir = config['data_dir']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    batch_size = config['batch_size']
    checkpoint_path = config['checkpoint']
    trainlist_path = config['trainlist']
    devlist_path = config['devlist']
    testlist_path = config['testlist']
    train_output_dir = config['train_output_dir']
    dev_output_dir = config['dev_output_dir']
    test_output_dir = config['test_output_dir']

    # ファイル名に学習率、エポック、バッチサイズを追加
    checkpoint_path = f'{checkpoint_path}_lr{learning_rate}_e{epochs}_bs{batch_size}.pth'
    logging.info("Checkpoint path set.")

    # ステージのマッピングS
    stages = ['feature_extraction', 'train', 'evaluate']
    start_stage = stages.index(args.start_stage)
    end_stage = stages.index(args.end_stage)
    logging.info("Stages mapped.")

    try:
        # 特徴量抽出ステージ
        if start_stage <= stages.index('feature_extraction') <= end_stage:
            logging.info("Starting feature extraction...")
            extract_features(config)
            logging.info("Feature extraction completed.")

        # 学習ステージ
        if start_stage <= stages.index('train') <= end_stage:
            logging.info("Starting training...")

            # データセットから特徴量の次元数を取得
            dataset = SpeechDataset(data_dir)
            example_speaker_id = 'M0101'  # 例として使用する話者コード（実際のデータに応じて変更）
            example_sentence_id = '001'  # 例として使用する発話番号（実際のデータに応じて変更）
            audio_dim, articulatory_dim, linguistic_dim = dataset.get_feature_dimensions(example_speaker_id, example_sentence_id)

            # ログに次元数を出力して確認
            logging.info(f"Audio Dimension: {audio_dim}, Articulatory Dimension: {articulatory_dim}, Linguistic Dimension: {linguistic_dim}")

            in_channels = articulatory_dim  # 入力チャンネル数を調音特徴量の次元数に設定
            input_dim = articulatory_dim + linguistic_dim  # 入力に調音特徴量と言語特徴量の次元を使用
            output_dim = audio_dim  # 出力に音響特徴量の次元を使用

            model = CNN_LSTM_WaveNet(
                in_channels=in_channels,  # 動的に設定
                cnn_channels=64,
                lstm_hidden=128,
                output_dim=audio_dim,
                wavenet_channels=128,
                embed_dim=linguistic_dim  # 言語特徴量の次元数を渡す
            ).to(device)
            logging.info("Model initialized.")

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            start_epoch = 0
            if checkpoint_path:
                start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
                logging.info("Checkpoint loaded.")

            if os.path.exists(checkpoint_path):
                start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
                logging.info("Checkpoint loaded.")
            else:
                logging.info(f"No checkpoint found at {checkpoint_path}. Starting from epoch 0.")

            train_model(
                model,
                optimizer,
                start_epoch=start_epoch,
                epochs=epochs,
                batch_size=batch_size,
                trainlist_path=trainlist_path,
                devlist_path=devlist_path,
                train_output_dir=train_output_dir,
                dev_output_dir=dev_output_dir,
                device=device,
                config=config  # config を渡す
            )
            logging.info("Training completed.")

        # 評価ステージ
        if start_stage <= stages.index('evaluate') <= end_stage:
            logging.info("Starting evaluation...")

            # データセットから特徴量の次元数を取得
            dataset = SpeechDataset(data_dir)
            example_speaker_id = 'M0101'  # 例として使用する話者コード
            example_sentence_id = '001'  # 例として使用する発話番号
            audio_dim, articulatory_dim, linguistic_dim = dataset.get_feature_dimensions(example_speaker_id, example_sentence_id)

            # ログに次元数を出力して確認
            logging.info(f"Audio Dimension: {audio_dim}, Articulatory Dimension: {articulatory_dim}, Linguistic Dimension: {linguistic_dim}")

            in_channels = articulatory_dim  # 入力チャンネル数を調音特徴量の次元数に設定

            model = CNN_LSTM_WaveNet(
                in_channels=in_channels,  # 動的に設定
                cnn_channels=64,
                lstm_hidden=128,
                output_dim=audio_dim,
                wavenet_channels=128,
                embed_dim=linguistic_dim  # 言語特徴量の次元数を渡す
            ).to(device)
            checkpoint_path = config['checkpoint']
            model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
            logging.info(f"Checkpoint loaded from {checkpoint_path}")
            model.eval()
            logging.info("Model loaded for evaluation.")

            test_items = load_testlist(testlist_path)
            metrics = evaluate_generated_audio(config)
            #logging.info(f"Metrics returned: {metrics}")
            #print_metrics(*metrics)
            logging.info("Evaluation completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Processing Pipeline")
    parser.add_argument("--start_stage", type=str, required=True, choices=['feature_extraction', 'train', 'evaluate'],
                        help="Stage of the pipeline to start execution")
    parser.add_argument("--end_stage", type=str, required=True, choices=['feature_extraction', 'train', 'evaluate'],
                        help="Stage of the pipeline to end execution")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file")

    args = parser.parse_args()
    main(args)
