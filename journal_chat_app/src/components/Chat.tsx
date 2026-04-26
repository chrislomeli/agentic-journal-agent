import { useEffect, useRef } from 'react'
import { Trash2 } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'

export function Chat() {
  const { messages, input, isLoading, error, handleInputChange, handleSubmit, stop, clearMessages } =
    useChat()

  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-col h-screen bg-slate-900 text-slate-100 font-sans">

      {/* Header */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <div>
          <h1 className="text-base font-medium tracking-tight text-slate-100">Journal Agent</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            {isLoading ? (
              <span className="text-indigo-400">Thinking…</span>
            ) : (
              'Ready'
            )}
          </p>
        </div>
        <button
          onClick={clearMessages}
          disabled={messages.length === 0}
          title="Clear conversation"
          className="
            p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800
            disabled:opacity-20 disabled:cursor-not-allowed transition-colors
          "
        >
          <Trash2 size={16} />
        </button>
      </header>

      {/* Message list */}
      <main className="flex-1 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            <div className="w-12 h-12 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
              <span className="text-lg">📓</span>
            </div>
            <p className="text-slate-500 text-sm max-w-xs">
              Start a conversation with your journal agent. Type a message below.
            </p>
          </div>
        ) : (
          <>
            {messages.map(message => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </>
        )}

        {/* Error banner */}
        {error && (
          <div className="mx-auto max-w-sm mt-4 px-4 py-3 bg-red-950 border border-red-800 rounded-xl text-red-300 text-xs text-center">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 px-6 py-4 border-t border-slate-800">
        <ChatInput
          value={input}
          onChange={handleInputChange}
          onSubmit={handleSubmit}
          onStop={stop}
          isLoading={isLoading}
        />
        <p className="text-center text-xs text-slate-600 mt-3">
          Enter to send · Shift+Enter for new line
        </p>
      </footer>
    </div>
  )
}
