import { useI18n } from "../components/I18nProvider";

const NOTEBOOKS = [
  {
    id: "transformer-practice",
    title: "Transformer 练习笔记本",
    title_en: "Transformer practice notebook",
    path: "notebook/practice1/transformer",
    note: "本地 Jupyter 练习，不是主工具台的替代。",
    note_en: "A local Jupyter extra. It does not replace the main tools.",
  },
];

export function NotebooksPage() {
  const { language, tr } = useI18n();
  return (
    <article className="mx-auto w-full max-w-[720px]">
      <p className="text-2xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {tr("额外练习", "Extras")}
      </p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">{tr("笔记本", "Notebooks")}</h1>
      <p className="mt-3 max-w-[65ch] text-sm leading-7 text-muted-foreground">
        {tr(
          "这些是额外练习，不替换工具台和实验室。图表、评测、注意力图仍在「实验室」和「工具台」。",
          "These are extras. Charts, evals, and attention maps still live in Labs and Tools.",
        )}
      </p>
      <ul className="mt-6 grid gap-3">
        {NOTEBOOKS.map((item) => (
          <li key={item.id} className="rounded-md border border-border px-4 py-3">
            <h2 className="text-sm font-semibold">{language === "en" ? item.title_en : item.title}</h2>
            <p className="mt-1 font-mono text-2xs text-muted-foreground">{item.path}</p>
            <p className="mt-2 text-sm text-muted-foreground">{language === "en" ? item.note_en : item.note}</p>
          </li>
        ))}
      </ul>
    </article>
  );
}
