import os
import sys
import json
import time
import subprocess
import tempfile
from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import whisper
import numpy as np
import torch
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from sklearn import __version__ as sklearn_version
from packaging import version
from tqdm import tqdm


# ナレーションキーワード定義
NARRATION_KEYWORDS = ["クトゥルフ神話TRPG", "はじめ"]


class AudioProcessor:
    """音声ファイルの処理を担当するクラス"""

    def __init__(self, target_sr: int = 16000):
        """
            target_sr: ターゲットサンプリングレート（Hz）
        """
        self.target_sr = target_sr

    # 音声ファイルを読み込む処理, 動画データの時は_load_via_ffmpeg()を実行
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """音声ファイルをロード

        Args:
            file_path: 音声/動画ファイルのパス

        Returns:
            (audio_array, sample_rate) のタプル
        """
        print(f"音声ファイルをロード中: {file_path}")

        try:
            # librosaで直接ロード
            audio, sr = librosa.load(
                file_path,
                sr=self.target_sr,
                mono=True
            )
            print(f"✓ librosaで正常にロード (sr: {sr}Hz)")
            return audio, sr    # pyright: ignore[reportReturnType]
        except Exception as e:
            print(f"✗ librosaでのロードに失敗: {e}")
            return self._load_via_ffmpeg(file_path)

    # データが音声ではなく動画だった場合の処理
    def _load_via_ffmpeg(self, file_path: str) -> Tuple[np.ndarray, int]:
        """FFmpegを使用して音声をロード（フォールバック）"""
        print("FFmpegで再抽出を試みます...")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "temp.wav")

            cmd = [
                "ffmpeg",
                "-i", file_path,
                "-ac", "1",  # モノラル
                "-ar", str(self.target_sr),
                "-hide_banner", "-loglevel", "error",
                wav_path
            ]

            try:
                subprocess.run(cmd, check=True)
                audio, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
                print(f"✓ FFmpegで正常にロード")
                return audio, sr    # pyright: ignore[reportReturnType]
            except Exception as e:
                raise RuntimeError(f"音声ロード失敗: {e}")

    # 音声をセグメントに分割する
    def split_into_segments(
        self,
        audio: np.ndarray,
        sr: int,
        segment_length: float = 20.0
    ) -> List[Dict]:
        """音声をセグメントに分割

        Args:
            audio: 音声配列
            sr: サンプリングレート
            segment_length: セグメント長（秒）

        Returns:
            セグメントリスト
        """
        print(f"音声を {segment_length}秒のセグメントに分割中...")

        segment_samples = int(segment_length * sr)
        segments = []

        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]

            # 1秒未満のセグメントは除外
            if len(segment) < sr:
                continue

            segments.append({
                "audio": segment,
                "start_time": i / sr,
                "end_time": (i + len(segment)) / sr
            })

        print(f"✓ {len(segments)}個のセグメントを生成")
        return segments


class TranscriptionProcessor:
    """文字起こし処理を担当するクラス"""

    def __init__(self, model_size: str = "medium"):
        """
        Args:
            model_size: Whisperモデルのサイズ（tiny/base/small/medium/large）
        """
        self.model_size = model_size
        self.model = None

    # 音声・動画データを文字起こし
    def transcribe(self, file_path: str) -> Dict:
        """Whisperで文字起こし

        Args:
            file_path: 音声/動画ファイルのパス

        Returns:
            文字起こし結果（辞書）
        """
        print(f"文字起こしを実行中 (モデル: {self.model_size})...")

        if self.model is None:
            self.model = whisper.load_model(self.model_size)

        options = {
            "language": "ja",
            "word_timestamps": True,
            "verbose": False
        }

        result = self.model.transcribe(file_path, **options)

        # GPU メモリ解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ 文字起こし完了")
        return result


class SpeakerDiarizer:
    """話者分離を担当するクラス"""

    def __init__(self):
        """初期化"""
        self.encoder = VoiceEncoder()

    # Whisperセグメント単位で話者特徴を抽出
    def extract_embeddings_per_whisper_segment(
        self,
        file_path: str,
        transcription: Dict,
        sr: int
    ) -> Tuple[List[int], np.ndarray]:
        """Whisper セグメントごとに埋め込みを抽出

        Args:
            file_path: 音声ファイルパス
            transcription: Whisper の文字起こし結果
            sr: サンプリングレート

        Returns:
            (セグメントインデックスリスト, 埋め込み行列)
        """
        print("Whisper セグメント単位で話者特徴を抽出中...")

        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        embeddings_list = []
        segment_indices = []

        # Whisper セグメント（会話の単位）ごとに処理
        for idx, seg in enumerate(tqdm(transcription["segments"])):
            duration = seg["end"] - seg["start"]

            # 0.5秒未満のセグメントはスキップ（信頼度が低い）
            if duration < 0.5:
                continue

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)

            # その時間範囲の音声だけを抽出
            segment_audio = audio[start_sample:end_sample]

            try:
                wav = preprocess_wav(segment_audio, source_sr=sr)
                embedding = self.encoder.embed_utterance(wav)
                embeddings_list.append(embedding)
                segment_indices.append(idx)
            except Exception as e:
                print(f"⚠ セグメント {idx} 処理エラー: {e}")
                # エラー時はゼロベクトルを追加
                embeddings_list.append(np.zeros(256))
                segment_indices.append(idx)

        # メモリ解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ {len(embeddings_list)}個の埋め込みを抽出")
        return segment_indices, np.array(embeddings_list)

    # 特徴量に基づいて話者をクラスタリング
    def cluster_speakers(
        self,
        embeddings: np.ndarray,
        max_speakers: int = 6
    ) -> np.ndarray:
        """話者クラスタリング

        Args:
            embeddings: 埋め込み行列
            max_speakers: 最大話者数

        Returns:
            話者ラベル配列
        """
        if len(embeddings) == 0:
            return np.array([])

        print(f"話者をクラスタリング中 (最大: {max_speakers})...")

        n_clusters = min(max_speakers, len(embeddings))

        # scikit-learnバージョン対応
        if version.parse(sklearn_version) >= version.parse("0.24.0"):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average"
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="average"
            )

        labels = clustering.fit_predict(embeddings)
        unique_speakers = len(np.unique(labels))
        print(f"✓ {unique_speakers}人の話者を検出")

        return labels


class TranscriptBuilder:
    """文字起こしデータを統合するクラス"""

    # Whisper セグメント単位での話者ラベリングに対応
    @staticmethod
    def build_transcript(
        transcription: Dict,
        segment_indices: List[int],
        speaker_labels: np.ndarray
    ) -> List[Dict]:
        """文字起こし結果と話者情報を統合（Whisper セグメント単位）

        Args:
            transcription: Whisper の文字起こし結果
            segment_indices: 処理済みセグメントのインデックスリスト
            speaker_labels: 話者ラベル配列

        Returns:
            統合トランスクリプト（リスト）
        """
        print("トランスクリプトを統合中...")

        # インデックスとラベルのマッピング
        speaker_map = {idx: label for idx, label in zip(segment_indices, speaker_labels)}

        # 一時的なエントリを生成
        temp_entries = []

        if "segments" in transcription:
            for idx, seg in enumerate(transcription["segments"]):
                text = seg["text"].strip()

                if not text:
                    continue

                # このセグメントに対応する話者ラベルを取得
                if idx in speaker_map:
                    speaker = f"SPEAKER_{speaker_map[idx]:02d}"
                else:
                    # 処理対象外のセグメント(0.5秒未満など)は前の話者を引き継ぐ
                    if temp_entries:
                        speaker = temp_entries[-1]["speaker"]
                    else:
                        speaker = "UNKNOWN"

                # タイムスタンプをフォーマット
                timestamp = str(timedelta(seconds=int(seg["start"]))).split('.')[0]

                temp_entries.append({
                    "timestamp": timestamp,
                    "speaker": speaker,
                    "content": text
                })

        # 同一話者の連続発言をマージ
        merged = []
        current_speaker = None
        current_entry = None

        for entry in temp_entries:
            if current_speaker != entry["speaker"]:
                if current_entry:
                    merged.append(current_entry)
                current_speaker = entry["speaker"]
                current_entry = entry.copy()
            else:
                current_entry["content"] += " " + entry["content"]  # pyright: ignore[reportOptionalSubscript]

        if current_entry:
            merged.append(current_entry)

        print(f"✓ {len(merged)}行のトランスクリプトを生成")
        return merged

    @staticmethod
    def detect_narration(transcript: List[Dict]) -> List[Dict]:
        """ナレーション自動判定

        Args:
            transcript: トランスクリプト

        Returns:
            is_narration フィールドが追加されたトランスクリプト
        """
        print("ナレーション判定中...")

        narrator_id = None

        for entry in transcript:
            entry["is_narration"] = False

            # ナレーターが未定の場合、キーワード検出
            if narrator_id is None:
                for keyword in NARRATION_KEYWORDS:
                    if keyword in entry["content"]:
                        narrator_id = entry["speaker"]
                        entry["is_narration"] = True
                        print(f"ナレーター検出: {narrator_id}")
                        break
            else:
                # 同一話者の発言にマークをつける
                if entry["speaker"] == narrator_id:
                    entry["is_narration"] = True

        return transcript

    @staticmethod
    def correct_transcript(transcript: List[Dict]) -> List[Dict]:
        """トランスクリプト校正

        Args:
            transcript: トランスクリプト

        Returns:
            校正済みトランスクリプト
        """
        for entry in transcript:
            # 不要な空白を削除
            content = entry["content"].replace("  ", " ")

            # 文末に句点がない場合は追加
            if content and not content.endswith(('。', '！', '？')):
                content += '。'

            entry["content"] = content

        return transcript


class TranscriptExporter:
    """トランスクリプトをJSONで出力するクラス"""

    @staticmethod
    def export_json(
        transcript: List[Dict],
        output_path: str,
        source_file: str,
        processing_time: float,
        num_speakers: int
    ) -> None:
        """JSONファイルに出力

        Args:
            transcript: トランスクリプト
            output_path: 出力ファイルパス
            source_file: ソースファイル名
            processing_time: 処理時間（秒）
            num_speakers: 話者数
        """
        data = {
            "metadata": {
                "source_file": source_file,
                "processing_time": round(processing_time, 2),
                "num_speakers": num_speakers,
                "created_at": datetime.now().isoformat()
            },
            "transcript": transcript
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ JSON出力: {output_path}")


def main(file_path: str, max_speakers: int = 6) -> str:
    """メイン処理

    Args:
        file_path: 入力ファイルパス
        max_speakers: 最大話者数

    Returns:
        出力JSONファイルパス
    """
    start_time = time.time()

    # 1. 音声をロード
    audio_proc = AudioProcessor()
    audio, sr = audio_proc.load_audio(file_path)

    # 2. 文字起こし
    trans_proc = TranscriptionProcessor()
    transcription = trans_proc.transcribe(file_path)

    # 3. Whisper セグメント単位で話者埋め込み抽出＆クラスタリング
    diarizer = SpeakerDiarizer()
    segment_indices, embeddings = diarizer.extract_embeddings_per_whisper_segment(
        file_path, transcription, sr
    )
    speaker_labels = diarizer.cluster_speakers(embeddings, max_speakers)

    # 4. トランスクリプト統合
    builder = TranscriptBuilder()
    transcript = builder.build_transcript(transcription, segment_indices, speaker_labels)
    transcript = builder.detect_narration(transcript)
    transcript = builder.correct_transcript(transcript)

    # 5. JSON出力
    processing_time = time.time() - start_time
    num_speakers = len(set(e["speaker"] for e in transcript))

    base, _ = os.path.splitext(file_path)
    output_path = base + ".json"

    exporter = TranscriptExporter()
    exporter.export_json(
        transcript,
        output_path,
        os.path.basename(file_path),
        processing_time,
        num_speakers
    )

    print(f"\n処理完了: {processing_time:.2f}秒")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_or_video_file> [max_speakers]")
        sys.exit(1)

    file_path = sys.argv[1]
    max_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    output_file = main(file_path, max_speakers)
    print(f"出力ファイル: {output_file}")