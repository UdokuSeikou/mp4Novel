import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { TranscriptEditor } from '../components/TranscriptEditor';

export const EditorPage: React.FC = () => {
	const { fileId } = useParams<{ fileId: string }>();
	const navigate = useNavigate();

	if (!fileId) {
		return <div>Invalid file ID</div>;
	}

	return (
		<TranscriptEditor
			fileId={fileId}
			onBack={() => navigate('/')}
		/>
	);
};
