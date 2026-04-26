import { useState, useCallback, useRef } from 'react'
import { Message } from '../types'

const API_URL = '/api/chat'  // proxied to http://localhost:8000/chat via vite.config.ts

function generateId(): string {
  return Math.random().toString(36).slice(2, 11)
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    setError(null)

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    // Optimistically add user message
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Placeholder for the streaming assistant message
    const assistantId = generateId()
    setMessages(prev => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', timestamp: new Date() },
    ])

    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          messages: [...messages, userMessage].map(m => ({
            role: m.role,
            content: m.content,
          })),
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      // ---------- SSE streaming ----------
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') break

            try {
              // Expect JSON: { "text": "..." }
              // Adjust this parsing to match whatever your FastAPI yields
              const parsed = JSON.parse(data)
              const text: string = parsed.text ?? parsed.content ?? data

              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + text }
                    : m
                )
              )
            } catch {
              // Raw text chunk (non-JSON SSE) — append as-is
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + data }
                    : m
                )
              )
            }
          }

          // Hook point: handle custom event types from your LangGraph nodes
          // e.g.  if (line.startsWith('event: node_update')) { ... }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return

      const message = (err as Error).message ?? 'Something went wrong'
      setError(message)

      // Remove the empty assistant placeholder on error
      setMessages(prev => prev.filter(m => m.id !== assistantId))
    } finally {
      setIsLoading(false)
    }
  }, [messages, isLoading])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value),
    []
  )

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      sendMessage(input)
    },
    [input, sendMessage]
  )

  const clearMessages = useCallback(() => setMessages([]), [])

  return {
    messages,
    input,
    isLoading,
    error,
    handleInputChange,
    handleSubmit,
    sendMessage,
    stop,
    clearMessages,
    setInput,
  }
}
