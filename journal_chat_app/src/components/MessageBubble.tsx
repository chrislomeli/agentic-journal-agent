import { Message } from '../types'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {!isUser && (
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center mr-2.5 mt-0.5">
          <span className="text-xs text-slate-300 font-mono font-medium">AI</span>
        </div>
      )}

      <div
        className={`
          max-w-[75%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed
          ${isUser
            ? 'bg-indigo-600 text-white rounded-tr-sm'
            : 'bg-slate-800 text-slate-100 rounded-tl-sm border border-slate-700'
          }
        `}
      >
        {message.content === '' && !isUser ? (
          // Typing indicator while streaming
          <span className="flex gap-1 items-center h-4">
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
          </span>
        ) : (
          <span className="whitespace-pre-wrap">{message.content}</span>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-indigo-500 flex items-center justify-center ml-2.5 mt-0.5">
          <span className="text-xs text-white font-mono font-medium">You</span>
        </div>
      )}
    </div>
  )
}
