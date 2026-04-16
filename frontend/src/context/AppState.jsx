import { useState } from 'react'
import AppStateContext from './appStateContext'

const STORAGE_KEY = 'docflow_state_v1'

function buildInvoiceFields(index) {
  const seed = index + 1
  return {
    invoiceNumber: `INV-20${24 + (seed % 3)}-${String(1000 + seed * 7)}`,
    invoiceDate: `2026-0${(seed % 8) + 1}-${String((seed % 27) + 1).padStart(2, '0')}`,
    dueDate: `2026-1${seed % 2}-${String((seed % 25) + 2).padStart(2, '0')}`,
    issuerName: `Issuer Group ${String.fromCharCode(65 + (seed % 5))}`,
    recipientName: `Recipient Ops ${String.fromCharCode(70 + (seed % 5))}`,
    totalAmount: `$${(320 + seed * 48.5).toFixed(2)}`,
  }
}

function classifyFiles(files) {
  const labels = ['invoice', 'form', 'resume', 'email', 'budget']
  return files.map((file, index) => {
    const label = labels[(file.name.length + index) % labels.length]
    return {
      fileName: file.name,
      fileSize: file.size,
      previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : '',
      label,
      confidence: 0.78 + ((index % 5) * 0.04),
      extraction: label === 'invoice' ? buildInvoiceFields(index) : null,
    }
  })
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

  const createClassificationJob = (files) => {
    const pages = classifyFiles(files)
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
