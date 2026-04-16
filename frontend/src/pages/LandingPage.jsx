import { Link } from 'react-router-dom'
import AppHeader from '../components/AppHeader'
import HeroVisual from '../components/HeroVisual'
import { useAppState } from '../context/useAppState'

function LandingPage() {
  const { isSignedIn } = useAppState()

  return (
    <div className="page-shell">
      <AppHeader />

      <main>
        <section className="hero-section">
          <div className="hero-copy reveal-up delay-2">
            <p className="eyebrow">MICROSOFT DiT DOCUMENT FLOW</p>
            <h1>Document intelligence.</h1>
            <p className="hero-description">
              Upload documents, classify pages, and review key invoice fields in a
              clean workflow.
            </p>
            <div className="hero-actions">
              <Link className="primary-button" to={isSignedIn ? '/app' : '/auth?mode=signin'}>
                {isSignedIn ? 'Start Classifying' : 'Sign In to Start'}
              </Link>
              <Link className="secondary-button" to="/auth?mode=signup">
                Create Account
              </Link>
            </div>
          </div>
          <HeroVisual />
        </section>
      </main>
    </div>
  )
}

export default LandingPage
