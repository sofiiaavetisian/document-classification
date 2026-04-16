import { Link, useNavigate } from 'react-router-dom'
import { useAppState } from '../context/useAppState'

function AppHeader() {
  const navigate = useNavigate()
  const { isSignedIn, signOut } = useAppState()

  return (
    <header className="top-nav reveal-up delay-1">
      <Link className="brand" to="/">
        <span className="brand-mark" aria-hidden="true"></span>
        DocFlow
      </Link>
      <div className="nav-actions">
        {!isSignedIn && (
          <>
            <Link className="ghost-button" to="/auth?mode=signin">
              Sign In
            </Link>
            <Link className="ghost-button" to="/auth?mode=signup">
              Sign Up
            </Link>
          </>
        )}
        {isSignedIn && (
          <>
            <Link className="ghost-button" to="/app">
              Open App
            </Link>
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                signOut()
                navigate('/')
              }}
            >
              Sign Out
            </button>
          </>
        )}
      </div>
    </header>
  )
}

export default AppHeader
