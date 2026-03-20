export interface ParsedMessage {
  thinking?: string;
  toolCalls: Array<{ name: string; arguments: any; raw: string }>;
  answer?: string;
  raw: string;
}

export function parseAssistantMessage(content: string): ParsedMessage {
  const result: ParsedMessage = {
    toolCalls: [],
    raw: content,
  };

  // 提取工具调用
  const toolCallRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
  let match;
  while ((match = toolCallRegex.exec(content)) !== null) {
    const toolCallStr = match[1].trim();
    try {
      const toolCall = JSON.parse(toolCallStr);
      result.toolCalls.push({
        name: toolCall.name || 'unknown',
        arguments: toolCall.arguments || {},
        raw: toolCallStr,
      });
    } catch {
      // 如果解析失败，保留原始字符串
      result.toolCalls.push({
        name: 'unknown',
        arguments: {},
        raw: toolCallStr,
      });
    }
  }

  // 提取最终答案
  const answerRegex = /<answer>([\s\S]*?)<\/answer>/;
  const answerMatch = content.match(answerRegex);
  if (answerMatch) {
    result.answer = answerMatch[1].trim();
  }

  // 提取思考过程（去除工具调用和答案后的内容）
  let thinking = content
    .replace(toolCallRegex, '')
    .replace(answerRegex, '')
    .trim();

  if (thinking) {
    result.thinking = thinking;
  }

  return result;
}

export function formatToolCall(toolCall: { name: string; arguments: any }): string {
  return JSON.stringify({ name: toolCall.name, arguments: toolCall.arguments }, null, 2);
}
