import axios from 'axios';
import type { ProgressMessage } from '../types';
import dotenv from 'dotenv';
dotenv.config();

const API_BASE_URL = process.env.API_BASE_URL;
const WS_BASE_URL = process.env.WS_BASE_URL;

export const apiClient = axios.create({
	baseURL: API_BASE_URL,
	headers: {
		'Content-Type': 'application/json',
	},
});

export const connectProgressWebSocket = (
	fileId: string,
	onProgress: (message: ProgressMessage) => void,
	onError: (error: Event) => void,
	onClose: () => void
): WebSocket => {
	const ws = new WebSocket(`${WS_BASE_URL}/ws/progress/${fileId}`);

	ws.onopen = () => {
		console.log('WebSocket connected');
	};

	ws.onmessage = (event) => {
		try {
			const data: ProgressMessage = JSON.parse(event.data);
			onProgress(data);
		} catch (e) {
			console.error('Failed to parse WebSocket message', e);
		}
	};

	ws.onerror = (error) => {
		console.error('WebSocket error', error);
		onError(error);
	};

	ws.onclose = () => {
		console.log('WebSocket closed');
		onClose();
	};

	return ws;
};

export const uploadFile = async (file: File, maxSpeakers: number = 6) => {
	const formData = new FormData();
	formData.append('file', file);

	// max_speakers is a query param in the backend implementation?
	// Checking app.py: async def transcribe(..., max_speakers: int = 6)
	// It's a query param (default behavior for non-path/non-body params in FastAPI)

	const response = await apiClient.post(`/api/transcribe?max_speakers=${maxSpeakers}`, formData, {
		headers: {
			'Content-Type': 'multipart/form-data',
		},
	});
	return response.data;
};

export const fetchTranscript = async (fileId: string) => {
	const response = await apiClient.get(`/api/transcript/${fileId}`);
	return response.data;
};

export const fetchTranscripts = async (skip: number = 0, limit: number = 10) => {
	const response = await apiClient.get(`/api/transcripts?skip=${skip}&limit=${limit}`);
	return response.data;
};

export const saveTranscript = async (fileId: string, transcript: any[]) => {
	const response = await apiClient.post(`/api/transcript/${fileId}/save`, { transcript });
	return response.data;
};

export const fetchSpeakers = async (fileId: string) => {
	const response = await apiClient.get(`/api/speakers/${fileId}`);
	return response.data;
};

export const deleteTranscript = async (fileId: string) => {
	const response = await apiClient.delete(`/api/transcript/${fileId}`);
	return response.data;
};
