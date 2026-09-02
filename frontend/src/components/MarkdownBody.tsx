import { useI18n } from "@/components/I18nProvider";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function MarkdownBody({ source }: { source: string }) {
  const { tr } = useI18n();
  if (!source.trim()) {
    return <p className="text-muted-foreground">{tr("这一页还没有正文。", "This page has no body yet.")}</p>;
  }
  return (
    <div className="prose prose-lesson max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{source}</ReactMarkdown>
    </div>
  );
}
