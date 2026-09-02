"use client";

import { useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab08DynamicTiling.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

type GridCandidate = {
  columns: number;
  rows: number;
  tiles: number;
  error: number;
};

function rankGrids(width: number, height: number, maxTiles: number) {
  const targetAspect = width / height;
  const candidates: GridCandidate[] = [];
  for (let rows = 1; rows <= maxTiles; rows += 1) {
    for (let columns = 1; columns <= maxTiles; columns += 1) {
      const tiles = rows * columns;
      if (tiles > maxTiles) continue;
      const error = Math.abs(Math.log(columns / rows / targetAspect));
      candidates.push({ columns, rows, tiles, error });
    }
  }
  return candidates.sort(
    (a, b) =>
      a.error - b.error ||
      b.tiles - a.tiles ||
      a.rows - b.rows ||
      a.columns - b.columns,
  );
}

export function Lab08DynamicTiling({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    imageWidth: initialNumber(initialState, "imageWidth", 1600),
    imageHeight: initialNumber(initialState, "imageHeight", 900),
    maxTiles: initialNumber(initialState, "maxTiles", 6),
    localRow: initialNumber(initialState, "localRow", 12),
    localColumn: initialNumber(initialState, "localColumn", 20),
  };
  const [imageWidth, setImageWidth] = useState(defaults.imageWidth);
  const [imageHeight, setImageHeight] = useState(defaults.imageHeight);
  const [maxTiles, setMaxTiles] = useState(defaults.maxTiles);
  const [includeThumbnail, setIncludeThumbnail] = useState(true);
  const [predictionColumns, setPredictionColumns] = useState("");
  const [predictionRows, setPredictionRows] = useState("");
  const [hasRun, setHasRun] = useState(false);
  const [selectedTile, setSelectedTile] = useState<number | null>(null);
  const [localRow, setLocalRow] = useState(defaults.localRow);
  const [localColumn, setLocalColumn] = useState(defaults.localColumn);
  const completedRef = useRef(false);

  const result = useMemo(() => {
    const ranked = rankGrids(imageWidth, imageHeight, maxTiles);
    const best = ranked[0];
    const patchesPerSide = 448 / 14;
    const tokensPerTile = patchesPerSide * patchesPerSide;
    const totalTokens =
      best.tiles * tokensPerTile + (includeThumbnail ? tokensPerTile : 0);
    return { ranked, best, patchesPerSide, tokensPerTile, totalTokens };
  }, [imageHeight, imageWidth, includeThumbnail, maxTiles]);

  const predictionCorrect =
    Number(predictionColumns) === result.best.columns &&
    Number(predictionRows) === result.best.rows;
  const gatePassed = hasRun && predictionCorrect && selectedTile !== null;

  const tileRow =
    selectedTile === null
      ? 0
      : Math.floor(selectedTile / result.best.columns);
  const tileColumn =
    selectedTile === null ? 0 : selectedTile % result.best.columns;
  const globalH = tileRow * result.patchesPerSide + localRow;
  const globalW = tileColumn * result.patchesPerSide + localColumn;

  function invalidate() {
    setHasRun(false);
    setSelectedTile(null);
    completedRef.current = false;
  }

  function runTiler() {
    setHasRun(true);
    setSelectedTile(null);
    completedRef.current = false;
  }

  function inspectTile(index: number) {
    setSelectedTile(index);
    if (predictionCorrect && !completedRef.current) {
      completedRef.current = true;
      const row = Math.floor(index / result.best.columns);
      const column = index % result.best.columns;
      onComplete?.({
        imageWidth,
        imageHeight,
        maxTiles,
        grid: [result.best.columns, result.best.rows],
        totalTokens: result.totalTokens,
        inspectedPosition: [
          0,
          row * result.patchesPerSide + localRow,
          column * result.patchesPerSide + localColumn,
        ],
      });
    }
  }

  function reset() {
    setImageWidth(defaults.imageWidth);
    setImageHeight(defaults.imageHeight);
    setMaxTiles(defaults.maxTiles);
    setIncludeThumbnail(true);
    setPredictionColumns("");
    setPredictionRows("");
    setHasRun(false);
    setSelectedTile(null);
    setLocalRow(defaults.localRow);
    setLocalColumn(defaults.localColumn);
    completedRef.current = false;
  }

  return (
    <section className={styles.lab} aria-labelledby="lab08-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>公式计算</span>
            <span>空间坐标</span>
          </div>
          <h3 id="lab08-title">一张长图怎样变成动态 Tile 与 M-RoPE 坐标？</h3>
          <p>
            先选最匹配宽高比的整数网格，再把 tile 内 patch 映射回全局
            (t, h, w)，把“看图”落到可计算的位置索引。
          </p>
        </div>
        <button type="button" className={styles.reset} onClick={reset}>
          重置画布
        </button>
      </header>

      <div className={styles.parameters}>
        <label>
          <span>
            图像宽度 <strong>{imageWidth}px</strong>
          </span>
          <input
            type="range"
            min="600"
            max="2400"
            step="100"
            value={imageWidth}
            onChange={(event) => {
              setImageWidth(Number(event.target.value));
              invalidate();
            }}
          />
        </label>
        <label>
          <span>
            图像高度 <strong>{imageHeight}px</strong>
          </span>
          <input
            type="range"
            min="600"
            max="2400"
            step="100"
            value={imageHeight}
            onChange={(event) => {
              setImageHeight(Number(event.target.value));
              invalidate();
            }}
          />
        </label>
        <label>
          <span>最大 tiles</span>
          <select
            value={maxTiles}
            onChange={(event) => {
              setMaxTiles(Number(event.target.value));
              invalidate();
            }}
          >
            {[4, 6, 8, 9].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={includeThumbnail}
            onChange={(event) => {
              setIncludeThumbnail(event.target.checked);
              invalidate();
            }}
          />
          <span>附加全局缩略图</span>
        </label>
      </div>

      <div className={styles.beforeAfter}>
        <div className={styles.sourcePanel}>
          <div className={styles.panelHead}>
            <span>原始画幅</span>
            <strong>
              {imageWidth} × {imageHeight}
            </strong>
          </div>
          <div
            className={styles.sourceImage}
            style={
              {
                "--source-aspect": `${imageWidth} / ${imageHeight}`,
              } as CSSProperties
            }
            role="img"
            aria-label={`宽高比 ${(imageWidth / imageHeight).toFixed(3)} 的教学图像`}
          >
            <div className={styles.horizon} aria-hidden="true" />
            <div className={styles.subject}>subject</div>
            <span>aspect = {(imageWidth / imageHeight).toFixed(3)}</span>
          </div>
          <code>
            score(c,r) = |ln((c/r) / (W/H))|
            <br />
            同分时使用更多 tiles
          </code>
        </div>

        <div className={styles.outputPanel}>
          <div className={styles.panelHead}>
            <span>动态切图结果</span>
            <strong>
              {hasRun
                ? `${result.best.columns} × ${result.best.rows}`
                : "? × ?"}
            </strong>
          </div>
          <div
            className={styles.tileCanvas}
            style={
              {
                "--tile-columns": hasRun ? result.best.columns : 2,
                "--tile-rows": hasRun ? result.best.rows : 2,
              } as CSSProperties
            }
          >
            {Array.from(
              {
                length: hasRun
                  ? result.best.tiles
                  : Math.min(4, maxTiles),
              },
              (_, index) => (
                <button
                  type="button"
                  key={index}
                  disabled={!hasRun}
                  aria-pressed={selectedTile === index}
                  onClick={() => inspectTile(index)}
                >
                  {hasRun ? (
                    <>
                      <strong>T{index}</strong>
                      <span>
                        r{Math.floor(index / result.best.columns)} c
                        {index % result.best.columns}
                      </span>
                    </>
                  ) : (
                    "?"
                  )}
                </button>
              ),
            )}
          </div>
          <div className={styles.tokenBill}>
            <span>视觉 token 账单</span>
            <strong>{hasRun ? result.totalTokens : "—"}</strong>
            <code>
              {hasRun
                ? `${result.best.tiles} × 32² ${
                    includeThumbnail ? "+ 1 × 32²" : ""
                  }`
                : "tiles × (448/14)²"}
            </code>
          </div>
        </div>
      </div>

      <div className={styles.predict}>
        <div>
          <span>先预测整数网格</span>
          <strong>
            在 tiles ≤ {maxTiles} 中，哪个 c × r 的宽高比误差最小？
          </strong>
        </div>
        <label>
          <span>c</span>
          <input
            type="number"
            min="1"
            max={maxTiles}
            value={predictionColumns}
            onChange={(event) => {
              setPredictionColumns(event.target.value);
              invalidate();
            }}
            aria-label="预测列数"
          />
        </label>
        <b>×</b>
        <label>
          <span>r</span>
          <input
            type="number"
            min="1"
            max={maxTiles}
            value={predictionRows}
            onChange={(event) => {
              setPredictionRows(event.target.value);
              invalidate();
            }}
            aria-label="预测行数"
          />
        </label>
        <button
          type="button"
          disabled={
            predictionColumns.trim() === "" || predictionRows.trim() === ""
          }
          onClick={runTiler}
        >
          运行 tiler
        </button>
      </div>

      {hasRun && (
        <div className={styles.coordinates} aria-live="polite">
          <div className={styles.ranking}>
            <span>候选网格（按公式排序）</span>
            <ol>
              {result.ranked.slice(0, 4).map((candidate) => (
                <li
                  key={`${candidate.columns}-${candidate.rows}`}
                  className={
                    candidate === result.best ? styles.best : undefined
                  }
                >
                  <b>
                    {candidate.columns} × {candidate.rows}
                  </b>
                  <span>{candidate.tiles} tiles</span>
                  <code>error {candidate.error.toFixed(4)}</code>
                </li>
              ))}
            </ol>
          </div>
          <div className={styles.mrope}>
            <div className={styles.mropeHead}>
              <div>
                <span>M-RoPE 坐标探针</span>
                <strong>
                  {selectedTile === null
                    ? "点击上方任一 tile"
                    : `T${selectedTile} · tile(${tileRow}, ${tileColumn})`}
                </strong>
              </div>
              <output>
                ({selectedTile === null ? "?" : 0},{" "}
                {selectedTile === null ? "?" : globalH},{" "}
                {selectedTile === null ? "?" : globalW})
              </output>
            </div>
            <div className={styles.patchControls}>
              <label>
                <span>
                  local h <b>{localRow}</b>
                </span>
                <input
                  type="range"
                  min="0"
                  max="31"
                  value={localRow}
                  onChange={(event) => setLocalRow(Number(event.target.value))}
                />
              </label>
              <label>
                <span>
                  local w <b>{localColumn}</b>
                </span>
                <input
                  type="range"
                  min="0"
                  max="31"
                  value={localColumn}
                  onChange={(event) =>
                    setLocalColumn(Number(event.target.value))
                  }
                />
              </label>
            </div>
            <code>
              global_h = tile_row × 32 + local_h
              <br />
              global_w = tile_col × 32 + local_w
            </code>
          </div>
        </div>
      )}

      <div
        className={`${styles.gate} ${
          hasRun
            ? gatePassed
              ? styles.pass
              : predictionCorrect
                ? styles.inspect
                : styles.retry
            : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "先预测 c × r；运行不会使用隐藏的模型分数。"
            : !predictionCorrect
              ? `预测不符。公式最优为 ${result.best.columns} × ${result.best.rows}，检查对数宽高比误差。`
              : selectedTile === null
                ? "网格预测正确。最后点击一个 tile，完成局部 patch → 全局 M-RoPE 坐标映射。"
                : "你已经完成从画幅、动态切图、token 账单到全局空间坐标的整条推导。"}
        </span>
      </div>
    </section>
  );
}
