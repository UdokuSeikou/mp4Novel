import React from 'react';
import '../styles/components/ColorPicker.css';

interface ColorPickerProps {
	color: string;
	onChange: (color: string) => void;
	presetColors?: string[];
}

const DEFAULT_PRESETS = [
	'#FF6B6B',
	'#4ECDC4',
	'#45B7D1',
	'#FFA07A',
	'#98D8C8',
	'#F7DC6F',
	'#BB8FCE',
	'#85C1E2',
	'#F8B739',
	'#52B788',
];

export const ColorPicker: React.FC<ColorPickerProps> = ({ color, onChange, presetColors = DEFAULT_PRESETS }) => {
	return (
		<div className="color-picker">
			<div
				className="color-preview"
				style={{ backgroundColor: color }}
			>
				<input
					type="color"
					value={color}
					onChange={(e) => onChange(e.target.value)}
					className="color-input"
				/>
			</div>
			<div className="preset-colors">
				{presetColors.map((preset) => (
					<button
						key={preset}
						className={`preset-btn ${color === preset ? 'active' : ''}`}
						style={{ backgroundColor: preset }}
						onClick={() => onChange(preset)}
						aria-label={`Select color ${preset}`}
					/>
				))}
			</div>
		</div>
	);
};
