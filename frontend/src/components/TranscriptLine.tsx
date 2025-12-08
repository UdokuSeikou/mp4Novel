import React, { useRef, useEffect } from 'react';
import { Mic, User } from 'lucide-react';
import type { TranscriptEntry, SpeakerInfo } from '../types';
import '../styles/components/TranscriptLine.css';

interface TranscriptLineProps {
	entry: TranscriptEntry;
	speakers: SpeakerInfo[];
	index: number;
	onUpdate: (index: number, updates: Partial<TranscriptEntry>) => void;
	onDelete: (index: number) => void;
	onFocus: (index: number) => void;
}

export const TranscriptLine: React.FC<TranscriptLineProps> = ({
	entry,
	speakers,
	index,
	onUpdate,
	onDelete,
	onFocus,
}) => {
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	// Auto-resize textarea
	useEffect(() => {
		if (textareaRef.current) {
			textareaRef.current.style.height = 'auto';
			textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
		}
	}, [entry.content]);

	// Find speaker info
	const speakerInfo = speakers.find((s) => s.speaker_id === entry.speaker);
	const speakerColor = speakerInfo?.color || '#ccc';
	// const speakerLabel = speakerInfo?.label || entry.speaker;

	return (
		<div className={`transcript-line ${entry.is_narration ? 'narration' : ''}`}>
			<div className="line-gutter">
				<span className="timestamp">{entry.timestamp}</span>
				<button
					className={`narration-toggle ${entry.is_narration ? 'active' : ''}`}
					onClick={() => onUpdate(index, { is_narration: !entry.is_narration })}
					title="Toggle Narration"
				>
					{entry.is_narration ? <Mic size={14} /> : <User size={14} />}
				</button>
			</div>

			<div className="line-content-wrapper">
				<div
					className="speaker-select-wrapper"
					style={{ borderLeftColor: speakerColor }}
				>
					<select
						value={entry.speaker}
						onChange={(e) => onUpdate(index, { speaker: e.target.value })}
						className="speaker-select"
						style={{ color: speakerColor }}
					>
						{speakers.map((s) => (
							<option
								key={s.speaker_id}
								value={s.speaker_id}
							>
								{s.label}
							</option>
						))}
					</select>
				</div>

				<textarea
					ref={textareaRef}
					value={entry.content}
					onChange={(e) => onUpdate(index, { content: e.target.value })}
					onFocus={() => onFocus(index)}
					className="content-textarea"
					rows={1}
				/>
			</div>

			<button
				onClick={() => onDelete(index)}
				className="delete-line-btn"
				title="Delete line"
			>
				×
			</button>
		</div>
	);
};
