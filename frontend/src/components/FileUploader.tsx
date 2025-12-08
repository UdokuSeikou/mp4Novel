import { useState, useCallback } from 'react';
import { Upload, FileVideo, AlertCircle } from 'lucide-react';
import { uploadFile, connectProgressWebSocket } from '../api/client';
import type { ProgressMessage, TranscribeResponse } from '../types';
import '../styles/components/FileUploader.css';

interface FileUploaderProps {
	onUploadComplete: (fileId: string) => void;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onUploadComplete }) => {
	const [isDragging, setIsDragging] = useState(false);
	const [file, setFile] = useState<File | null>(null);
	const [uploading, setUploading] = useState(false);
	const [progress, setProgress] = useState<number>(0);
	const [statusMessage, setStatusMessage] = useState<string>('');
	const [error, setError] = useState<string | null>(null);

	const handleDrag = useCallback((e: React.DragEvent) => {
		e.preventDefault();
		e.stopPropagation();
		if (e.type === 'dragenter' || e.type === 'dragover') {
			setIsDragging(true);
		} else if (e.type === 'dragleave') {
			setIsDragging(false);
		}
	}, []);

	const handleDrop = useCallback((e: React.DragEvent) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(false);

		if (e.dataTransfer.files && e.dataTransfer.files[0]) {
			handleFileSelect(e.dataTransfer.files[0]);
		}
	}, []);

	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files && e.target.files[0]) {
			handleFileSelect(e.target.files[0]);
		}
	};

	const handleFileSelect = (selectedFile: File) => {
		// Validate file type
		const allowedTypes = ['video/mp4', 'audio/mpeg', 'audio/wav', 'audio/x-m4a', 'video/quicktime'];
		// Note: 'video/quicktime' is for .mov

		if (!allowedTypes.includes(selectedFile.type) && !selectedFile.name.endsWith('.mov')) {
			setError('Invalid file type. Please upload MP4, MP3, WAV, M4A, or MOV.');
			return;
		}

		setFile(selectedFile);
		setError(null);
	};

	const startUpload = async () => {
		if (!file) return;

		setUploading(true);
		setProgress(0);
		setStatusMessage('Uploading file...');
		setError(null);

		try {
			// 1. Upload File
			const response: TranscribeResponse = await uploadFile(file);
			const fileId = response.file_id;

			setStatusMessage('File uploaded. processing...');

			// 2. Connect WebSocket
			const ws = connectProgressWebSocket(
				fileId,
				(msg: ProgressMessage) => {
					setProgress(msg.progress);
					setStatusMessage(msg.message);

					if (msg.step === 'completed') {
						onUploadComplete(fileId);
						setUploading(false); // Should convert to "Completed" state or redirect
						ws.close();
					} else if (msg.step === 'error') {
						setError(msg.error || 'Unknown error occurred');
						setUploading(false);
						ws.close();
					}
				},
				() => {
					// Error handling is done in onerror
				},
				() => {
					// Close handling
				}
			);
		} catch (err: any) {
			console.error(err);
			setError(err.response?.data?.detail?.error || 'Upload failed');
			setUploading(false);
		}
	};

	return (
		<div className="file-uploader-container">
			{!uploading ? (
				<div
					className={`dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
					onDragEnter={handleDrag}
					onDragLeave={handleDrag}
					onDragOver={handleDrag}
					onDrop={handleDrop}
				>
					<input
						type="file"
						id="file-input"
						className="file-input"
						onChange={handleChange}
						accept=".mp4,.mp3,.wav,.m4a,.mov"
					/>

					<div className="dropzone-content">
						{file ? (
							<>
								<FileVideo
									size={48}
									className="icon-primary"
								/>
								<p className="file-name">{file.name}</p>
								<p className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
								<button
									onClick={startUpload}
									className="upload-btn"
								>
									Start Transcription
								</button>
								<button
									onClick={() => setFile(null)}
									className="clear-btn"
								>
									Choose another file
								</button>
							</>
						) : (
							<>
								<Upload
									size={48}
									className="icon-secondary"
								/>
								<p>Drag & Drop files here</p>
								<p className="sub-text">or</p>
								<label
									htmlFor="file-input"
									className="browse-btn"
								>
									Browse Files
								</label>
								<p className="file-types">Supported: MP4, MP3, WAV, M4A, MOV</p>
							</>
						)}
					</div>
				</div>
			) : (
				<div className="progress-container card">
					<div className="progress-header">
						<h3>Transcribing...</h3>
						<span className="progress-percent">{progress}%</span>
					</div>
					<div className="progress-bar-bg">
						<div
							className="progress-bar-fill"
							style={{ width: `${progress}%` }}
						/>
					</div>
					<p className="status-message">{statusMessage}</p>
				</div>
			)}

			{error && (
				<div className="error-message">
					<AlertCircle size={20} />
					<span>{error}</span>
				</div>
			)}
		</div>
	);
};
