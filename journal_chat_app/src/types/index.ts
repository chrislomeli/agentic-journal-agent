export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface ChatRequest {
  messages: { role: string; content: string }[]
  session_id?: string
}
