import { useRef, useEffect, KeyboardEvent } from 'react'
import { Send, Square } from 'lucide-react'

interface ChatInputProps {
  value: string
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  onSubmit: (e: React.FormEvent) => void
  onStop: () => void
  isLoading: boolean
  disabled?: boolean
}

export function ChatInput({ value, onChange, onSubmit, onStop, isLoading }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!isLoading && value.trim()) {
        onSubmit(e as unknown as React.FormEvent)
      }
    }
  }

  return (
    <form onSubmit={onSubmit} className="flex items-end gap-3">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={onChange}
        onKeyDown={handleKeyDown}
        placeholder="Message the agent… (Shift+Enter for new line)"
        rows={1}
        className="
          flex-1 resize-none bg-slate-800 border border-slate-700 rounded-xl
          px-4 py-3 text-sm text-slate-100 placeholder-slate-500
          focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500
          transition-colors font-sans leading-relaxed
        "
      />
      {isLoading ? (
        <button
          type="button"
          onClick={onStop}
          className="
            flex-shrink-0 w-10 h-10 rounded-xl bg-slate-700 hover:bg-red-600
            flex items-center justify-center transition-colors
          "
          title="Stop generating"
        >
          <Square size={16} className="text-slate-300" />
        </button>
      ) : (
        <button
          type="submit"
          disabled={!value.trim()}
          className="
            flex-shrink-0 w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500
            disabled:opacity-30 disabled:cursor-not-allowed
            flex items-center justify-center transition-colors
          "
          title="Send (Enter)"
        >
          <Send size={16} className="text-white" />
        </button>
      )}
    </form>
  )
}
