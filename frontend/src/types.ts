export interface TranscriptEntry {
	timestamp: string;
	speaker: string;
	content: string;
	is_narration: boolean;
	color?: string; // Frontend only, derived from speaker info
}

export interface TranscriptMetadata {
	source_file: string;
	processing_time: number;
	num_speakers: number;
	created_at: string;
}

export interface TranscriptResponse {
	file_id: string;
	metadata: TranscriptMetadata;
	transcript: TranscriptEntry[];
}

export interface SpeakerInfo {
	speaker_id: string;
	label: string;
	is_narration: boolean;
	color: string;
}

export interface SpeakersResponse {
	speakers: SpeakerInfo[];
}

export interface TranscribeResponse {
	file_id: string;
	filename: string;
	status: string;
	message: string;
}

export interface ProgressMessage {
	step: 'loading_audio' | 'transcribing' | 'extracting_speakers' | 'clustering' | 'saving' | 'completed' | 'error';
	progress: number;
	message: string;
	result?: TranscriptResponse;
	error?: string;
}
