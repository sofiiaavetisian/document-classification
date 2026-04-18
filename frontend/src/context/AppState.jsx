import { useState } from 'react'
import AppStateContext from './appStateContext'

const STORAGE_KEY = 'docflow_state_v1'

// Base URL for the FastAPI backend. Update this if the backend runs on a
// different host or port.
const API_BASE = 'http://localhost:8000'

/**
 * Map the backend snake_case fields dict to the camelCase extraction shape
 * expected by AnalysisPage.
 */
function mapFields(fields) {
  if (!fields) return null
  const pick = (snake, upper) => fields[snake] || fields[upper] || ''
  return {
    invoiceNumber: pick('invoice_number', 'INVOICE_NUMBER'),
    invoiceDate:   pick('invoice_date',   'INVOICE_DATE'),
    dueDate:       pick('due_date',       'DUE_DATE'),
    issuerName:    pick('issuer_name',    'ISSUER_NAME'),
    recipientName: pick('recipient_name', 'RECIPIENT_NAME'),
    totalAmount:   pick('total_amount',   'TOTAL_AMOUNT'),
  }
}

/**
 * Call POST /predict for a single file and return a page-shaped object.
 * Falls back to an error page shape if the request fails.
 */
async function classifyFile(file) {
  const previewUrl = file.type.startsWith('image/') ? URL.createObjectURL(file) : ''

  const form = new FormData()
  form.append('file', file)

  try {
    const res  = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form })
    const data = await res.json()

    if (data.error) {
      return {
        fileName:   file.name,
        fileSize:   file.size,
        previewUrl,
        label:      'error',
        confidence: 0,
        extraction: null,
        error:      data.error,
      }
    }

    return {
      fileName:   file.name,
      fileSize:   file.size,
      previewUrl,
      label:      data.predicted_class || 'unknown',
      confidence: data.confidence      ?? 0,
      extraction: mapFields(data.fields),
      processingMode: data.processing_mode || '',
      backendError: data.error || null,
    }
  } catch (err) {
    return {
      fileName:   file.name,
      fileSize:   file.size,
      previewUrl,
      label:      'error',
      confidence: 0,
      extraction: null,
      error:      `Network error: ${err.message}`,
      processingMode: '',
      backendError: null,
    }
  }
}

function loadInitialState() {
  const fallback = { isSignedIn: false, history: [] }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback
  } catch {
    return fallback
  }
}

function AppStateProvider({ children }) {
  const initial = loadInitialState()
  const [isSignedIn, setIsSignedIn] = useState(initial.isSignedIn)
  const [history, setHistory] = useState(initial.history)

  const persist = (nextIsSignedIn, nextHistory) => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ isSignedIn: nextIsSignedIn, history: nextHistory }),
    )
  }

  const signIn = () => {
    setIsSignedIn(true)
    persist(true, history)
  }

  const signOut = () => {
    setIsSignedIn(false)
    persist(false, history)
  }

  const createClassificationJob = async (files) => {
    const pages = await Promise.all(files.map((file) => classifyFile(file)))
    const invoicePages = pages.filter((page) => page.label === 'invoice').length
    const job = {
      id: `job_${Date.now()}`,
      createdAt: new Date().toISOString(),
      totalFiles: files.length,
      invoicePages,
      pages,
    }
    const nextHistory = [job, ...history]
    setHistory(nextHistory)
    persist(isSignedIn, nextHistory)
    return job
  }

  const getJobById = (jobId) => history.find((job) => job.id === jobId) ?? null

  const value = {
    isSignedIn,
    history,
    signIn,
    signOut,
    createClassificationJob,
    getJobById,
  }

  return (
    <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
  )
}

export default AppStateProvider
