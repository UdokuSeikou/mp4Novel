import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ListPage } from './pages/ListPage';
import { EditorPage } from './pages/EditorPage';

function App() {
	return (
		<Router>
			<Routes>
				<Route
					path="/"
					element={<ListPage />}
				/>
				<Route
					path="/editor/:fileId"
					element={<EditorPage />}
				/>
			</Routes>
		</Router>
	);
}

export default App;
