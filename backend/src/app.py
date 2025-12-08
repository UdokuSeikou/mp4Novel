"""
mp4Novel FastAPI バックエンド

動画・音声ファイルから話者分離付き文字起こしを行うAPI
"""

import os
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List
from enum import Enum
from datetime import datetime

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from transcribe import main as transcribe_main, TranscriptBuilder


# ==================== 設定 ====================
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.joinpath(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    UPLOAD_FOLDER: Path = Path("./default")
    OUTPUT_FOLDER: Path = Path("./default")
    INT_VALUE: int = -1


settings = Settings()

UPLOAD_FOLDER = settings.UPLOAD_FOLDER
OUTPUT_FOLDER = settings.OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".mov"}

# ディレクトリ作成
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ==================== Pydantic モデル ====================

class ProgressEvent(str, Enum):
    """進捗イベントの種類"""
    LOADING_AUDIO = "loading_audio"
    TRANSCRIBING = "transcribing"
    EXTRACTING_SPEAKERS = "extracting_speakers"
    CLUSTERING = "clustering"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"


class TranscriptEntry(BaseModel):
    """トランスクリプト行"""
    timestamp: str
    speaker: str
    content: str
    is_narration: bool
    color: Optional[str] = None


class TranscriptMetadata(BaseModel):
    """メタデータ"""
    source_file: str
    processing_time: float
    num_speakers: int
    created_at: str


class TranscriptResponse(BaseModel):
    """トランスクリプト取得レスポンス"""
    file_id: str
    metadata: TranscriptMetadata
    transcript: List[TranscriptEntry]


class ProgressMessage(BaseModel):
    """進捗通知メッセージ"""
    step: ProgressEvent
    progress: int
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None


class SpeakerInfo(BaseModel):
    """話者情報"""
    speaker_id: str
    label: str
    is_narration: bool
    color: str


class SpeakersResponse(BaseModel):
    """話者情報取得レスポンス"""
    speakers: List[SpeakerInfo]


class TranscribeResponse(BaseModel):
    """文字起こし開始レスポンス"""
    file_id: str
    filename: str
    status: str
    message: str


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str
    error_code: str
    status_code: int


# ==================== グローバル状態 ====================

# ファイルID → 進捗情報のマッピング
progress_map: dict[str, dict] = {}

# WebSocket接続の管理
class ConnectionManager:
    """WebSocket接続管理"""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, file_id: str, websocket: WebSocket):
        await websocket.accept()
        if file_id not in self.active_connections:
            self.active_connections[file_id] = []
        self.active_connections[file_id].append(websocket)

    def disconnect(self, file_id: str, websocket: WebSocket):
        self.active_connections[file_id].remove(websocket)
        if not self.active_connections[file_id]:
            del self.active_connections[file_id]

    async def broadcast(self, file_id: str, message: dict):
        if file_id in self.active_connections:
            for connection in self.active_connections[file_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"⚠ WebSocket送信エラー: {e}")


manager = ConnectionManager()


# ==================== FastAPI アプリケーション ====================

app = FastAPI(
    title="mp4Novel API", description="動画・音声ファイル文字起こしAPI", version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では ["http://localhost:3000"] など指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ヘルパー関数 ====================


def is_allowed_file(filename: str | None) -> bool:
    """許可されたファイル形式かチェック"""
    if filename is not None:
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
    return False


async def update_progress(
    file_id: str,
    step: ProgressEvent,
    progress: int,
    message: str,
    result: dict | None = None,
    error: str | None = None,
):
    """進捗情報を更新してWebSocketで配信"""
    progress_map[file_id] = {
        "step": step,
        "progress": progress,
        "message": message,
        "result": result,
        "error": error,
    }

    await manager.broadcast(
        file_id,
        {
            "step": step.value,
            "progress": progress,
            "message": message,
            "result": result,
            "error": error,
        },
    )


async def process_transcription(file_path: str, file_id: str, max_speakers: int = 6):
    """バックグラウンドで文字起こし処理を実行"""
    try:
        # メイン処理を実行（非同期で実行）
        loop = asyncio.get_running_loop()

        def callback(step: str, progress: int, message: str):
            # Enum変換を試みる
            try:
                event = ProgressEvent(step)
            except ValueError:
                event = ProgressEvent.TRANSCRIBING # フォールバック

            asyncio.run_coroutine_threadsafe(
                update_progress(file_id, event, progress, message),
                loop
            )

        output_json = await asyncio.to_thread(transcribe_main, file_path, max_speakers, callback)

        # 生成されたJSONを読み込む
        with open(output_json, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        # 結果をoutputs フォルダに移動
        output_path = OUTPUT_FOLDER / f"{file_id}.json"
        import shutil

        shutil.move(output_json, output_path)

        # 完了
        await update_progress(
            file_id,
            ProgressEvent.COMPLETED,
            100,
            "処理完了",
            result={
                "file_id": file_id,
                "filename": Path(file_path).name,
                "json_path": str(output_path),
                "processing_time": result_data["metadata"]["processing_time"],
                "num_speakers": result_data["metadata"]["num_speakers"],
                "transcript": result_data["transcript"],
            },
        )

        # テンポラリファイルを削除
        os.remove(file_path)

    except Exception as e:
        print(f"✗ 処理エラー: {e}")
        await update_progress(
            file_id,
            ProgressEvent.ERROR,
            -1,
            "処理中にエラーが発生しました",
            error=str(e),
        )


def get_speakers_with_colors(file_id: str) -> List[SpeakerInfo]:
    """話者情報を色付きで取得"""
    # プリセットカラーパレット
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#FFA07A",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E2",
        "#F8B739",
        "#52B788",
    ]

    # JSON ファイルから読み込む
    json_path = OUTPUT_FOLDER / f"{file_id}.json"
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ユニークな話者を抽出
    speakers = {}
    for entry in data["transcript"]:
        speaker_id = entry["speaker"]
        if speaker_id not in speakers:
            speaker_idx = len(speakers)
            speakers[speaker_id] = {
                "speaker_id": speaker_id,
                "label": f"Speaker {speaker_idx + 1}",
                "is_narration": entry["is_narration"],
                "color": colors[speaker_idx % len(colors)],
            }

    return [SpeakerInfo(**info) for info in speakers.values()]


# ==================== エンドポイント ====================


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "mp4Novel API",
        "version": "1.0.0",
        "description": "動画・音声ファイル文字起こしAPI",
    }


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_speakers: int = 6,
):
    """
    ファイルアップロード + 文字起こし開始

    - file: MP4/MP3/WAV/M4A ファイル
    - max_speakers: 最大話者数（デフォルト: 6）

    Returns:
        - file_id: 処理ID（進捗通知の際に使用）
        - filename: アップロードされたファイル名
        - status: 処理ステータス（processing）
        - message: メッセージ
    """

    # ファイル形式チェック
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file format. MP4, MP3, WAV, M4A only.",
                "error_code": "INVALID_FILE_FORMAT",
                "status_code": 400,
            },
        )

    # ファイルID生成
    file_id = str(uuid.uuid4())

    # 一時的に保存
    file_path = UPLOAD_FOLDER / f"{file_id}_{file.filename}"
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # バックグラウンドタスク登録
    if background_tasks is not None:
        background_tasks.add_task(
            process_transcription, str(file_path), file_id, max_speakers
        )

    return TranscribeResponse(
        file_id=file_id,
        filename=file.filename, # pyright: ignore[reportArgumentType]
        status="processing",
        message="文字起こしを開始しました",
    )


@app.websocket("/ws/progress/{file_id}")
async def websocket_progress(websocket: WebSocket, file_id: str):
    """
    WebSocket: 進捗通知

    クライアントが接続すると、バックグラウンドタスクの進捗がリアルタイムで配信されます
    """
    await manager.connect(file_id, websocket)

    try:
        # 既に進捗情報がある場合は最新情報を送信
        if file_id in progress_map:
            info = progress_map[file_id]
            await websocket.send_json(
                {
                    "step": info["step"].value,
                    "progress": info["progress"],
                    "message": info["message"],
                    "result": info["result"],
                    "error": info["error"],
                }
            )

        # クライアントからのメッセージ受信（キープアライブ用）
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(file_id, websocket)
    except Exception as e:
        print(f"⚠ WebSocketエラー: {e}")
        manager.disconnect(file_id, websocket)


@app.get("/api/transcript/{file_id}", response_model=TranscriptResponse)
async def get_transcript(file_id: str):
    """
    トランスクリプト取得

    Args:
        file_id: 処理ID

    Returns:
        - file_id: 処理ID
        - metadata: メタデータ（処理時間、話者数など）
        - transcript: トランスクリプト配列
    """
    json_path = OUTPUT_FOLDER / f"{file_id}.json"

    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Transcript not found",
                "error_code": "NOT_FOUND",
                "status_code": 404,
            },
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pydanticモデルに変換
    entries = [
        TranscriptEntry(**entry, color=None)  # 初期は色なし（フロントで設定）
        for entry in data["transcript"]
    ]

    return TranscriptResponse(
        file_id=file_id,
        metadata=TranscriptMetadata(**data["metadata"]),
        transcript=entries,
    )


@app.post("/api/transcript/{file_id}/save")
async def save_transcript(file_id: str, data: dict):
    """
    トランスクリプト編集結果を保存

    Args:
        file_id: 処理ID
        data: 編集済みトランスクリプト

    Returns:
        - status: 保存ステータス
        - edited_json_path: 編集済みJSONのパス
        - message: メッセージ
    """
    json_path = OUTPUT_FOLDER / f"{file_id}.json"

    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Transcript not found",
                "error_code": "NOT_FOUND",
                "status_code": 404,
            },
        )

    # 元のメタデータを保持
    with open(json_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    # 編集済みファイルを保存
    edited_data = {
        "metadata": original["metadata"],
        "transcript": data.get("transcript", original["transcript"]),
    }

    edited_path = OUTPUT_FOLDER / f"{file_id}.edited.json"
    with open(edited_path, "w", encoding="utf-8") as f:
        json.dump(edited_data, f, ensure_ascii=False, indent=2)

    return {
        "status": "saved",
        "edited_json_path": str(edited_path),
        "message": "編集内容を保存しました",
    }


@app.get("/api/transcripts")
async def list_transcripts(skip: int = 0, limit: int = 10):
    """
    トランスクリプト一覧取得

    Args:
        skip: スキップ件数
        limit: 取得件数

    Returns:
        - items: トランスクリプト情報配列
        - total: 全体件数
        - skip: スキップ件数
        - limit: 取得件数
    """
    # JSON ファイル一覧を取得
    json_files = sorted(
        [
            f
            for f in OUTPUT_FOLDER.glob("*.json")
            if not f.name.endswith(".edited.json")
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    items = []
    for json_file in json_files[skip : skip + limit]:
        file_id = json_file.stem
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 編集済みファイルの有無をチェック
        edited_path = OUTPUT_FOLDER / f"{file_id}.edited.json"

        items.append(
            {
                "file_id": file_id,
                "filename": data["metadata"]["source_file"],
                "created_at": data["metadata"]["created_at"],
                "num_speakers": data["metadata"]["num_speakers"],
                "processing_time": data["metadata"]["processing_time"],
                "has_edited": edited_path.exists(),
            }
        )

    return {"items": items, "total": len(json_files), "skip": skip, "limit": limit}


@app.delete("/api/transcript/{file_id}")
async def delete_transcript(file_id: str):
    """
    トランスクリプト削除

    Args:
        file_id: 処理ID

    Returns:
        - status: 削除ステータス
        - file_id: 削除された処理ID
    """
    json_path = OUTPUT_FOLDER / f"{file_id}.json"
    edited_path = OUTPUT_FOLDER / f"{file_id}.edited.json"

    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Transcript not found",
                "error_code": "NOT_FOUND",
                "status_code": 404,
            },
        )

    # JSON ファイル削除
    os.remove(json_path)

    # 編集済みファイルがあれば削除
    if edited_path.exists():
        os.remove(edited_path)

    # 進捗情報をクリア
    if file_id in progress_map:
        del progress_map[file_id]

    return {"status": "deleted", "file_id": file_id}


@app.get("/api/speakers/{file_id}", response_model=SpeakersResponse)
async def get_speakers(file_id: str):
    """
    話者情報取得

    Args:
        file_id: 処理ID

    Returns:
        - speakers: 話者情報配列
    """
    speakers = get_speakers_with_colors(file_id)

    if not speakers:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Speakers not found",
                "error_code": "NOT_FOUND",
                "status_code": 404,
            },
        )

    return SpeakersResponse(speakers=speakers)


# ==================== メイン ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
