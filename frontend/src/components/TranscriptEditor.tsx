import React, { useState, useEffect } from 'react';
import { Save, ArrowLeft } from 'lucide-react';
import { TranscriptLine } from './TranscriptLine';
import { ColorPicker } from './ColorPicker';
import { fetchTranscript, saveTranscript, fetchSpeakers } from '../api/client';
import type { TranscriptEntry, SpeakerInfo, TranscriptMetadata } from '../types';
import '../styles/components/TranscriptEditor.css';

interface TranscriptEditorProps {
	fileId: string;
	onBack: () => void;
}

export const TranscriptEditor: React.FC<TranscriptEditorProps> = ({ fileId, onBack }) => {
	const [loading, setLoading] = useState(true);
	const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
	const [metadata, setMetadata] = useState<TranscriptMetadata | null>(null);
	const [speakers, setSpeakers] = useState<SpeakerInfo[]>([]);
	const [saving, setSaving] = useState(false);
	const [lastSaved, setLastSaved] = useState<Date | null>(null);

	useEffect(() => {
		loadData();
	}, [fileId]);

	const loadData = async () => {
		try {
			setLoading(true);
			const [transcriptData, speakersData] = await Promise.all([
				fetchTranscript(fileId),
				fetchSpeakers(fileId), // This is implied to be needed for consistent speaker colors/labels
			]);

			setTranscript(transcriptData.transcript);
			setMetadata(transcriptData.metadata);
			setSpeakers(speakersData.speakers);
			setLoading(false);
		} catch (err) {
			console.error('Failed to load transcript', err);
			setLoading(false);
		}
	};

	const handleUpdateLine = (index: number, updates: Partial<TranscriptEntry>) => {
		const newTranscript = [...transcript];
		newTranscript[index] = { ...newTranscript[index], ...updates };
		setTranscript(newTranscript);
	};

	const handleDeleteLine = (index: number) => {
		if (window.confirm('Are you sure you want to delete this line?')) {
			const newTranscript = [...transcript];
			newTranscript.splice(index, 1);
			setTranscript(newTranscript);
		}
	};

	const handleSpeakerUpdate = (speakerId: string, updates: Partial<SpeakerInfo>) => {
		const newSpeakers = speakers.map((s) => (s.speaker_id === speakerId ? { ...s, ...updates } : s));
		setSpeakers(newSpeakers);
	};

	const handleSave = async () => {
		try {
			setSaving(true);

			await saveTranscript(fileId, transcript);
			setLastSaved(new Date());
			setSaving(false);
		} catch (err) {
			console.error('Failed to save', err);
			setSaving(false);
			alert('Failed to save changes');
		}
	};

	if (loading) return <div className="loading-spinner">Loading transcript...</div>;

	return (
		<div className="editor-container">
			<header className="editor-header">
				<div className="header-left">
					<button
						onClick={onBack}
						className="back-btn"
					>
						<ArrowLeft size={20} />
					</button>
					<div>
						<h1>{metadata?.source_file}</h1>
						<p className="metadata-info">
							{metadata?.num_speakers} Speakers • {Math.round(metadata?.processing_time || 0)}s processing
							time
						</p>
					</div>
				</div>

				<div className="header-right">
					{lastSaved && <span className="last-saved">Saved {lastSaved.toLocaleTimeString()}</span>}
					<button
						onClick={handleSave}
						disabled={saving}
						className="save-btn"
					>
						<Save size={18} />
						{saving ? 'Saving...' : 'Save Changes'}
					</button>
				</div>
			</header>

			<div className="editor-layout">
				<aside className="speakers-panel">
					<h3>Speakers</h3>
					<div className="speakers-list">
						{speakers.map((speaker) => (
							<div
								key={speaker.speaker_id}
								className="speaker-item card"
							>
								<div className="speaker-header">
									<span className="speaker-label">{speaker.label}</span>
									{speaker.is_narration && <span className="tag-narration">Narrator</span>}
								</div>
								<div className="speaker-controls">
									<ColorPicker
										color={speaker.color}
										onChange={(c) => handleSpeakerUpdate(speaker.speaker_id, { color: c })}
									/>
									<input
										type="text"
										value={speaker.label}
										onChange={(e) =>
											handleSpeakerUpdate(speaker.speaker_id, { label: e.target.value })
										}
										className="speaker-name-input"
									/>
								</div>
							</div>
						))}
					</div>
				</aside>

				<main className="transcript-content card">
					{transcript.map((entry, idx) => (
						<TranscriptLine
							key={idx}
							index={idx}
							entry={entry}
							speakers={speakers}
							onUpdate={handleUpdateLine}
							onDelete={handleDeleteLine}
							onFocus={() => {}}
						/>
					))}

					{transcript.length === 0 && <div className="empty-state">No transcript data.</div>}
				</main>
			</div>
		</div>
	);
};
