# MegaCP Paper Materials

本目录保存 MegaCP 论文的写作蓝图和可编辑方法图，不包含运行时代码修改。

## 内容

- [`MEGACP_PAPER_WRITING_FRAMEWORK.md`](MEGACP_PAPER_WRITING_FRAMEWORK.md)：中文论文写作框架、技术细节、公式、实验问题、消融、证据索引和 claim audit。
- `figures/src/*.dot`：Graphviz 可编辑图源。
- `figures/*.svg`：适合 Markdown 和矢量编辑的导出图。
- `figures/*.pdf`：适合 LaTeX/论文排版的矢量导出图。
- `figures/*.png`：用于快速预览和视觉检查的高分辨率位图。
- [`scripts/render_figures.sh`](scripts/render_figures.sh)：从 `.dot` 源文件重新生成三种格式。

## 重新渲染

在仓库根目录执行：

```bash
bash paper/scripts/render_figures.sh
```

脚本只依赖 Graphviz 的 `dot` 命令。图中文字使用英文，正文蓝图使用中文；这样图稿可以直接迁移到英文论文排版中。

## 图稿清单

| Figure | 内容 |
| --- | --- |
| Fig. 1 | MegaCP 共同根问题、共同抽象和两个阶段实例 |
| Fig. 2 | operator-centric execution 与 tile-level dependency execution 对比 |
| Fig. 3 | MegaRing forward 的 K/V ingress、readiness、segment fusion 和 role reuse |
| Fig. 4 | MegaRing backward core 的 step-local dKV 与 owner-directed reduce-add |
| Fig. 5 | W=8 Buddy-Ring hierarchy 和 BR-PBS 搜索流程 |
| Fig. 6 | MegaDCP 的异构 task graph、dependency state 和 post-Q policy |
| Fig. 7 | NoSplit、OverSplit 与 Critical-Wave 的 CTA wave 对比 |

## 证据边界

论文蓝图将内容标记为 `CODE FACT`、`DESIGN RATIONALE`、`MEASURED EVIDENCE` 或 `VALIDATION TODO`。仓库已有 benchmark 结果保留其原始 timing boundary；在统一受控重测之前，不应把不同口径的结果拼成摘要 headline。
