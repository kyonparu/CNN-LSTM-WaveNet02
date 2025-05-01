import os
import numpy as np
import pandas as pd
import pyworld as pw
import pysptk
from tqdm import tqdm
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from nnmnkwii.preprocessing.f0 import interp1d
from utils import extract_speaker_and_sentence_id
from scipy.interpolate import UnivariateSpline

# 音素埋め込み用の辞書（例）
PHONEME_LIST = ["a", "i", "u", "e", "o", "k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p", "N", "silB", "silE", "sp"]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}

class PhonemeEmbedding(nn.Module):
    def __init__(self, num_phonemes, embedding_dim):
        super(PhonemeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_phonemes, embedding_dim)
    
    def forward(self, phoneme_ids):
        return self.embedding(phoneme_ids)

def f0_to_lf0(f0):
    """
    基本周波数(f0)を対数基本周波数(lf0)に変換する関数
    :param f0: 基本周波数
    :return: 対数基本周波数
    """
    lf0 = np.copy(f0)
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices] + 1e-6)
    return lf0

def compute_dynamic_features(features, window_static, window_first, window_second):
    """
    音声特徴量の動的特徴量（一次微分、二次微分）を計算し、全特徴量に統合する
    :param features: 基本的な音声特徴量（リストで渡される）
    :param window_static: 静的特徴量の窓
    :param window_first: 一次微分特徴量の窓
    :param window_second: 二次微分特徴量の窓
    :return: 動的特徴量を追加した音声特徴量
    """
    combined_features = []
    for feature in features:
        static_feature = np.apply_along_axis(lambda x: np.convolve(x, window_static, mode='same'), axis=0, arr=feature)
        combined_features.append(static_feature)
        first_diff = np.apply_along_axis(lambda x: np.convolve(x, window_first, mode='same'), axis=0, arr=feature)
        combined_features.append(first_diff)
        second_diff = np.apply_along_axis(lambda x: np.convolve(x, window_second, mode='same'), axis=0, arr=feature)
        combined_features.append(second_diff)
    return np.concatenate(combined_features, axis=-1)

def interpolate_f0(f0, timeaxis, target_frame_period):
    """
    基本周波数(f0)を指定されたフレームシフトで補完する関数
    :param f0: 基本周波数
    :param timeaxis: 時間軸
    :param target_frame_period: 目標フレームシフト (秒)
    :return: 補完された基本周波数と新しい時間軸
    """
    # スプライン補完
    spline = UnivariateSpline(timeaxis, f0, s=0, k=3)

    # 新しい時間軸を生成
    new_timeaxis = np.arange(timeaxis[0], timeaxis[-1], target_frame_period)

    # 補完されたf0を計算
    f0_interpolated = spline(new_timeaxis)

    return f0_interpolated, new_timeaxis

class SpeechDataset:
    def __init__(self, data_dir, embedding_dim=8):
        """
        コンストラクタ
        :param data_dir: データディレクトリのパス
        :param embedding_dim: 音素埋め込みの次元数
        """
        self.data_dir = data_dir
        self.phoneme_embedding = PhonemeEmbedding(len(PHONEME_LIST), embedding_dim)
        self.scaler = StandardScaler()  # StandardScalerのインスタンスを作成

    def load_articulation_data(self, speaker_id, sentence_id):
        """
        調音位置データの読み込みと一次微分値の計算
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 特徴量 (N, 6, 6) の NumPy 配列
        """
        features = []
        positions = ["UL", "LL", "LJ", "T1", "T2", "T3"]
        for pos in positions:
            file_name = f"ATR503{speaker_id}_{sentence_id}_{pos}.csv"
            file_path = os.path.join(self.data_dir, 'articulatory_data', speaker_id, file_name)
            data = pd.read_csv(file_path, header=None).values
            coords = data[:, :3]
            diff_coords = np.diff(coords, axis=0, prepend=coords[:1])
            combined = np.concatenate([coords, diff_coords], axis=1)
            features.append(combined)
        
        return np.stack(features, axis=1)

    def load_audio_features(self, speaker_id, sentence_id, target_length):
        """
        音声特徴量の抽出（pyworldを使用）
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :param target_length: 目標の長さ（articulation_featuresの長さ）
        :return: 特徴量辞書
        """
        file_name = f"ATR503{speaker_id}_{sentence_id}.wav"
        file_path = os.path.join(self.data_dir, 'audio_data', speaker_id, file_name)
        
        try:
            waveform, sample_rate = sf.read(file_path)
        except Exception as e:
            raise

        try:
            # 100Hzのフレームシフトを使用してf0を計算
            f0, timeaxis = pw.dio(waveform, sample_rate, frame_period=10.0)
            
            # 250Hzに補完
            target_frame_period = 1.0 / 250.0  # 250Hz のシフト
            f0_interpolated, new_timeaxis = interpolate_f0(f0, timeaxis, target_frame_period)
            
            # f0_interpolated のゼロまたは負の値を小さな正の値に置き換える
            f0_interpolated[f0_interpolated <= 0] = 1e-10
            
            # 対数基本周波数
            lf0 = np.log(f0_interpolated)
            
            # 線形補完と平滑化
            clf0 = interp1d(lf0, kind="linear")
            
            # 帯域非周期性指標
            try:
                aperiodicity = pw.d4c(waveform, f0_interpolated, new_timeaxis, sample_rate)
            except Exception as e:
                raise

            try:
                bap = pw.code_aperiodicity(aperiodicity, sample_rate)
            except Exception as e:
                raise

            # メルケプストラム
            spectrogram = pw.cheaptrick(waveform, f0_interpolated, new_timeaxis, sample_rate)
            mgc_order = 79
            alpha = pysptk.util.mcepalpha(sample_rate)
            mgc = pysptk.sp2mc(spectrogram, mgc_order, alpha)

            # 有声/無声フラグ
            voiced_flag = (f0_interpolated > 0).astype(np.float32)

            # 基本周波数と有声/無声フラグを2次元の行列の形にしておく
            lf0 = lf0[:, np.newaxis] if len(lf0.shape) == 1 else lf0
            voiced_flag = voiced_flag[:, np.newaxis] if len(voiced_flag.shape) == 1 else voiced_flag
            clf0 = clf0[:, np.newaxis]

            # 音響特徴量の長さを調整
            def pad_to_match_length(feature, target_length):
                """
                特徴量の長さを指定された長さに合わせるために前後に値を追加する関数
                :param feature: 特徴量 (2D NumPy 配列)
                :param target_length: 目標の長さ
                :return: 長さを調整した特徴量
                """
                current_length = feature.shape[0]
                if current_length >= target_length:
                    return feature

                diff = target_length - current_length
                pad_before = diff // 2
                pad_after = diff - pad_before

                return np.pad(feature, ((pad_before, pad_after), (0, 0)), mode='edge')

            # 音響特徴量の前後に値を動的に追加
            clf0 = pad_to_match_length(clf0, target_length=target_length)
            mgc = pad_to_match_length(mgc, target_length=target_length)
            bap = pad_to_match_length(bap, target_length=target_length)
            voiced_flag = pad_to_match_length(voiced_flag, target_length=target_length)

            return {
                "log_f0": clf0,
                "spectral_envelope": mgc,
                "aperiodicity": bap,
                "voiced_flag": voiced_flag
            }
        except Exception as e:
            raise

    
    def load_lab_file(self, speaker_id, sentence_id):
        """
        ラベルファイルの読み込み
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 音素ラベルのリスト
        """
        file_name = f"ATR503{speaker_id}_{sentence_id}.csv"
        file_path = os.path.join(self.data_dir, 'label_data_250', speaker_id, file_name)
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                phoneme = line.strip()
                phoneme_id = PHONEME_TO_ID.get(phoneme, 0)
                labels.append(phoneme_id)

        return torch.tensor(labels, dtype=torch.long)

    def __call__(self, speaker_id, sentence_id):
        """
        データローダーを呼び出して音声と調音位置の特徴量を取得
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 音声特徴量, 調音位置特徴量, 言語特徴量
        """
        # 調音特徴量の読み込み
        articulation_features = self.load_articulation_data(speaker_id, sentence_id)

        # 音声特徴量の読み込み
        audio_features = self.load_audio_features(speaker_id, sentence_id, target_length=articulation_features.shape[0])

        # 言語特徴量の読み込み
        phoneme_ids = self.load_lab_file(speaker_id, sentence_id)
        phoneme_embeddings = self.phoneme_embedding(phoneme_ids)
        phoneme_embeddings = phoneme_embeddings.clone().detach().float()

        # 音響特徴量の結合
        window_static = np.array([1])
        window_first = np.array([-0.5, 0.0, 0.5])
        window_second = np.array([1.0, -2.0, 1.0])

        feats = np.hstack([
            audio_features["spectral_envelope"],  # mgc
            audio_features["log_f0"],             # lf0
            audio_features["voiced_flag"],        # vuv
            audio_features["aperiodicity"]        # bap
        ])
        combined_audio_features = compute_dynamic_features([feats], window_static, window_first, window_second)

        # 音響特徴量の正規化
        combined_audio_features = self.scaler.fit_transform(combined_audio_features)

        # NumPy配列をテンソルに変換
        combined_audio_features = torch.tensor(combined_audio_features, dtype=torch.float32).clone().detach()
        articulation_features = torch.tensor(articulation_features, dtype=torch.float32).clone().detach()
        phoneme_embeddings = torch.tensor(phoneme_embeddings, dtype=torch.float32).clone().detach().float()

        # 特徴量の型を表示
        print(f"combined_audio_features type: {type(combined_audio_features)}, shape: {combined_audio_features.shape}")
        print(f"articulation_features type: {type(articulation_features)}, shape: {articulation_features.shape}")
        print(f"phoneme_embeddings type: {type(phoneme_embeddings)}, shape: {phoneme_embeddings.shape}")

        return combined_audio_features, articulation_features, phoneme_embeddings

    def get_feature_dimensions(self, speaker_id, sentence_id):
        """
        音響特徴量、調音特徴量、言語特徴量の次元数を取得する関数
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 音響特徴量の次元数, 調音特徴量の次元数, 言語特徴量の次元数
        """
        combined_audio_features, combined_articulatory_features, combined_linguistic_features = self(speaker_id, sentence_id)
        audio_dim = combined_audio_features.shape[1]
        articulatory_dim = combined_articulatory_features.shape[1]
        linguistic_dim = combined_linguistic_features.shape[1]
        print(f"Audio Dimension: {audio_dim}, Articulatory Dimension: {articulatory_dim}, Linguistic Dimension: {linguistic_dim}")
        # デバッグ用に次元数を表示
        return audio_dim, articulatory_dim, linguistic_dim


def extract_features(config):
    """
    特徴量抽出を行う関数
    :param config: 設定ファイルの情報を含む辞書
    """
    dataset = SpeechDataset(data_dir=config['data_dir'])
    
    # 出力ディレクトリの設定
    output_dirs = {
        'train': config['train_output_dir'],
        'dev': config['dev_output_dir'],
        'test': config['test_output_dir']
    }

    norm_dirs = {key: os.path.join(output_dirs[key], 'norm', key) for key in output_dirs}
    orig_dirs = {key: os.path.join(output_dirs[key], 'orig', key) for key in output_dirs}

    # ディレクトリ構造を作成
    for dirs in [norm_dirs, orig_dirs]:
        for key in dirs:
            os.makedirs(os.path.join(dirs[key], 'in'), exist_ok=True)
            os.makedirs(os.path.join(dirs[key], 'out'), exist_ok=True)
    
    # 特徴量抽出と保存
    for list_name, list_file in zip(['train', 'dev', 'test'], [config['trainlist'], config['devlist'], config['testlist']]):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        # tqdmを使って進捗バーを表示
        for line in tqdm(lines, desc=f"Extracting features for {list_name}"):
            item = line.strip()
            speaker_id, sentence_id = extract_speaker_and_sentence_id(item)
            if speaker_id is None or sentence_id is None:
                continue

            # lstm_features を削除
            combined_audio_features, articulation_features, phoneme_embeddings = dataset(speaker_id, sentence_id)

            # 正規化データの保存
            norm_output_dir = norm_dirs[list_name]
            torch.save(combined_audio_features, os.path.join(norm_output_dir, 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"))
            torch.save(articulation_features, os.path.join(norm_output_dir, 'out', f"articulation_features_{speaker_id}_{sentence_id}.pt"))  # [time_steps, 6, 6]
            torch.save(phoneme_embeddings, os.path.join(norm_output_dir, 'out', f"phoneme_embeddings_{speaker_id}_{sentence_id}.pt"))

            # 元データの保存
            orig_output_dir = orig_dirs[list_name]
            torch.save(combined_audio_features, os.path.join(orig_output_dir, 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"))
            torch.save(articulation_features, os.path.join(orig_output_dir, 'out', f"articulation_features_{speaker_id}_{sentence_id}.pt"))  # [time_steps, 6, 6]
            torch.save(phoneme_embeddings, os.path.join(orig_output_dir, 'out', f"phoneme_embeddings_{speaker_id}_{sentence_id}.pt"))

def load_saved_features(speaker_id, sentence_id, output_dir, list_name):
    """
    保存されたテンソルをロードする関数
    :param speaker_id: 話者コード
    :param sentence_id: 発話番号
    :param output_dir: 特徴量が保存されているベースディレクトリ (例: dump/)
    :param list_name: データセットの種類（train, dev, test）
    :return: ロードされたテンソル
    """
    norm_output_dir = os.path.join(output_dir, 'norm', list_name)
    combined_audio_features = torch.load(os.path.join(norm_output_dir, 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"), weights_only=True)
    articulation_features = torch.load(os.path.join(norm_output_dir, 'out', f"articulation_features_{speaker_id}_{sentence_id}.pt"), weights_only=True)
    phoneme_embeddings = torch.load(os.path.join(norm_output_dir, 'out', f"phoneme_embeddings_{speaker_id}_{sentence_id}.pt"), weights_only=True)

    # 3つの値を返す
    return combined_audio_features, articulation_features, phoneme_embeddings