import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.use({
  breaks: true,
  gfm: true,
});

export function renderMarkdown(text: string) {
  const raw = text ?? '';
  const html = marked.parse(raw);
  return DOMPurify.sanitize(html);
}
