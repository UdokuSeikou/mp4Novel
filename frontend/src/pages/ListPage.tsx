import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Clock, Trash2, Edit } from 'lucide-react';
import { FileUploader } from '../components/FileUploader';
import { fetchTranscripts, deleteTranscript } from '../api/client';
import type { TranscriptMetadata } from '../types';
import '../styles/pages/ListPage.css';

interface TranscriptItem extends TranscriptMetadata {
	file_id: string;
	has_edited: boolean;
}

export const ListPage: React.FC = () => {
	const navigate = useNavigate();
	const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		loadTranscripts();
	}, []);

	const loadTranscripts = async () => {
		try {
			setLoading(true);
			const data = await fetchTranscripts();
			setTranscripts(data.items);
			setLoading(false);
		} catch (err) {
			console.error('Failed to load transcripts', err);
			setLoading(false);
		}
	};

	const handleUploadComplete = (fileId: string) => {
		// Navigate to editor directly after upload
		navigate(`/editor/${fileId}`);
	};

	const handleDelete = async (e: React.MouseEvent, fileId: string) => {
		e.stopPropagation();
		if (window.confirm('Are you sure you want to delete this transcript?')) {
			try {
				await deleteTranscript(fileId);
				loadTranscripts();
			} catch (err) {
				console.error('Failed to delete', err);
			}
		}
	};

	return (
		<div className="list-page">
			<h1>mp4Novel</h1>

			<section className="upload-section">
				<FileUploader onUploadComplete={handleUploadComplete} />
			</section>

			<section className="list-section">
				<h2>Your Transcripts</h2>
				{loading ? (
					<p>Loading...</p>
				) : (
					<div className="transcript-grid">
						{transcripts.map((item) => (
							<div
								key={item.file_id}
								className="transcript-card card"
								onClick={() => navigate(`/editor/${item.file_id}`)}
							>
								<div className="card-header">
									<FileText
										className="icon-doc"
										size={24}
									/>
									{item.has_edited && <span className="edited-badge">Edited</span>}
								</div>
								<h3>{item.source_file}</h3>
								<div className="card-meta">
									<span className="meta-item">
										<Clock size={14} />
										{new Date(item.created_at).toLocaleDateString()}
									</span>
									<span className="meta-item">{item.num_speakers} Speakers</span>
								</div>
								<div className="card-actions">
									<button
										className="action-btn"
										title="Edit"
									>
										<Edit size={16} />
									</button>
									<button
										className="action-btn delete"
										onClick={(e) => handleDelete(e, item.file_id)}
										title="Delete"
									>
										<Trash2 size={16} />
									</button>
								</div>
							</div>
						))}

						{!loading && transcripts.length === 0 && (
							<p className="no-data">No transcripts yet. Upload a video to get started.</p>
						)}
					</div>
				)}
			</section>
		</div>
	);
};
