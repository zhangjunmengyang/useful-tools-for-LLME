import { useSearchParams } from "react-router-dom";

import { ToolsExplorer } from "./ToolsExplorer";

export function ToolsPage() {
  const [params] = useSearchParams();
  return <ToolsExplorer initialToolId={params.get("tool") ?? undefined} />;
}
