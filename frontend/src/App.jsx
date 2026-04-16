import { Navigate, Route, Routes } from 'react-router-dom'
import './App.css'
import { useAppState } from './context/useAppState'
import AnalysisPage from './pages/AnalysisPage'
import AuthPage from './pages/AuthPage'
import LandingPage from './pages/LandingPage'
import UploadPage from './pages/UploadPage'

function App() {
  const { isSignedIn } = useAppState()

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/auth" element={<AuthPage />} />
      <Route
        path="/app"
        element={isSignedIn ? <UploadPage /> : <Navigate to="/auth" replace />}
      />
      <Route
        path="/analysis/:jobId"
        element={isSignedIn ? <AnalysisPage /> : <Navigate to="/auth" replace />}
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
