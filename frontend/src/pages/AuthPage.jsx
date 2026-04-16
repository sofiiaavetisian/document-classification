import { useMemo } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import AppHeader from '../components/AppHeader'
import { useAppState } from '../context/useAppState'

function AuthPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { signIn } = useAppState()

  const mode = useMemo(() => {
    const value = new URLSearchParams(location.search).get('mode')
    return value === 'signup' ? 'signup' : 'signin'
  }, [location.search])

  return (
    <div className="page-shell auth-page">
      <AppHeader />

      <main className="auth-main reveal-up delay-2">
        <section className="auth-card">
          <p className="eyebrow">ACCOUNT ACCESS</p>
          <h2>{mode === 'signup' ? 'Create your account' : 'Sign in to DocFlow'}</h2>
          <p className="auth-copy">
            {mode === 'signup'
              ? 'Create your workspace to run DiT document classification and extraction.'
              : 'Continue to your classifier workspace and recent analysis history.'}
          </p>

          <form
            className="auth-form"
            onSubmit={(event) => {
              event.preventDefault()
              signIn()
              navigate('/app')
            }}
          >
            <label>
              Email
              <input type="email" placeholder="name@school.edu" required />
            </label>
            <label>
              Password
              <input type="password" placeholder="••••••••" required />
            </label>
            {mode === 'signup' && (
              <label>
                Confirm Password
                <input type="password" placeholder="••••••••" required />
              </label>
            )}

            <button className="auth-submit" type="submit">
              {mode === 'signup' ? 'Sign Up and Continue' : 'Sign In and Continue'}
            </button>
          </form>

          <p className="auth-switch">
            {mode === 'signup' ? 'Already have an account?' : "Don't have an account?"}{' '}
            <Link to={mode === 'signup' ? '/auth?mode=signin' : '/auth?mode=signup'}>
              {mode === 'signup' ? 'Sign in here' : 'Create one'}
            </Link>
          </p>
        </section>
      </main>
    </div>
  )
}

export default AuthPage
