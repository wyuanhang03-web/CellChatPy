"""Generate a readable research report from recorded evidence, never estimates."""
from pathlib import Path
import json
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def median(records, field):
    values = [r[field] for r in records if r.get('status') == 'completed' and field in r]
    return statistics.median(values) if values else None


def number(value, decimals=2):
    return f'{value:,.{decimals}f}' if value is not None else '未完成'


def tutorial_inference_time(label, directory):
    path = ROOT / 'results/tutorials' / label / directory / 'evidence/inference_calls.json'
    if not path.exists():
        return None
    calls = json.loads(path.read_text(encoding='utf-8'))
    return sum(row['elapsed_seconds'] for row in calls) if calls else None


def main():
    lines = ['# 实验结果：保持 CellChat 定义的大规模低内存推断', '',
             '本报告仅汇总保存的执行记录。推断核时间不包含导入、数据准备、差异基因筛选、绘图或下游系统分析。', '',
             '基线：GitHub `986bf7ef909f1e15059aad8a9a9ee31ac675ffc3`；开发版：`1.1.0+streaming.20261020`。', '',
             '## 回归测试', '']
    xml = ROOT / 'validation/combined_regression.xml'
    if xml.exists():
        tree = ET.parse(xml).getroot()
        suites = list(tree.iter('testsuite'))
        totals = {key: sum(int(x.attrib.get(key, 0)) for x in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        lines += [f"共 {totals['tests']} 项：失败 {totals['failures']}，错误 {totals['errors']}，跳过 {totals['skipped']}。",
                  '证据：[JUnit](validation/combined_regression.xml)、[执行前后源码及测试指纹](validation/regression_provenance.json)。', '']
    else:
        lines += ['尚无最终 JUnit 记录。', '']
    lines += ['## 七个公开教程', '']
    comparison = ROOT / 'validation/tutorial_comparison_reference_development.json'
    if comparison.exists():
        report = json.loads(comparison.read_text(encoding='utf-8'))
        lines += [f"完整七本对照通过：**{report.get('passed', False)}**。",
                  '完整 notebook 墙钟时间含导入、绘图、审计导出和未优化的网络分析。',
                  '群体推断列另列记录器捕获的 compute_communication_probability 调用耗时总和，单位秒。', '',
                  '| 教程 | 原版执行 | 开发版执行 | 原版总秒 | 开发版总秒 | 群体推断秒 原/新 | 数值对照 |',
                  '|---|---|---|---:|---:|---:|---|']
        for row in report['tutorials']:
            directory = row['directory']
            t0, t1 = tutorial_inference_time('reference', directory), tutorial_inference_time('development', directory)
            inference = f'{number(t0)} / {number(t1)}' if t0 is not None or t1 is not None else '不适用（spot 流程）'
            lines.append(f"| {directory} | {row.get('reference_status','缺失')} | {row.get('candidate_status','缺失')} | "
                         f"{number(row.get('reference_seconds'))} | {number(row.get('candidate_seconds'))} | {inference} | "
                         f"{'通过' if row.get('passed') else '未通过'} |")
        lines += ['', '概率报告最大误差；P 值要求逐项精确相等，显著性使用原来的严格 `< 0.05`。',
                  '比较导出的 LR 推断网络及 count/weight 聚合结果；路径分析等原始代码单元完整执行。',
                  '保存图文件比较记录；随机布局的像素位置不作为核心数值一致性的判据。',
                  '原公开 notebook 与 GitHub 源码代码单元一致。Visium 手算断言的一个阈值边界修复对两引擎同样应用，',
                  '详见 [修订记录](validation/tutorial_compatibility_patches.json)。', '']
        figures = [(row['directory'], item) for row in report['tutorials']
                   for item in row.get('figure_comparisons', [])]
        lines += [f"每版保存 {len(figures)} 张图，其中 {sum(item['sha256_equal'] for _, item in figures)} 张文件 SHA-256 相同。"]
        for directory, item in figures:
            if not item['sha256_equal']:
                lines.append(f"文件不同：`{directory}/{item['figure']}`。")
        if (ROOT / 'validation/figure_difference_notes.md').exists():
            lines += ['唯一不同气泡图的列顺序及标签对应关系见 [图像差异核查](validation/figure_difference_notes.md)。']
        lines += ['']
    else:
        lines += ['完整教程对照尚未生成。请查看各运行目录中的 summary.json；不能把未完成标记为通过。', '']
    saved_objects = ROOT / 'validation/tutorial_saved_object_comparison.json'
    if saved_objects.exists():
        saved_report = json.loads(saved_objects.read_text(encoding='utf-8'))
        lines += ['### 本次运行保存对象的补充对照', '',
                  f"四组保存对象补充检查通过：**{saved_report.get('passed', False)}**。",
                  '从本次 notebook 新生成的 pickle 中，分别用对应版本导出网络、路径网络及可用的 spot 网络。',
                  '比较包括名称、数值、中心性等已保存字段；稀疏矩阵按原生 data/indices/indptr 比较，不展开为密集矩阵。', '',
                  '| 教程 | 比较数组数 | 逐元素相等数组数 | 最大绝对误差 | 结果 |',
                  '|---|---:|---:|---:|---|']
        for row in saved_report['tutorials']:
            arrays = row['arrays']
            maximum = max((a.get('max_abs_error', 0.0) for a in arrays), default=0.0)
            lines.append(f"| {row['directory']} | {len(arrays)} | "
                         f"{sum(bool(a.get('bitwise_equal')) for a in arrays)} | {maximum:.3g} | "
                         f"{'通过' if row['passed'] else '未通过'} |")
        different = [(row['directory'], a) for row in saved_report['tutorials']
                     for a in row['arrays'] if not a.get('bitwise_equal')]
        scopes = sorted({directory + ': ' + '/'.join(a['path'].split('/')[1:3])
                         for directory, a in different})
        if different:
            fields = sorted({a['path'].split('/')[-1] for _, a in different})
            lines += ['', f"非精确相等数组共 {len(different)} 项，范围为 {'；'.join(scopes)}，字段为 {', '.join(fields)}。",
                      '这些差异均通过事先设定的浮点检查（绝对、相对容差各 1e-12），没有为通过实验放宽门槛。',
                      '每个字段的误差见 [差异附表](validation/tutorial_tolerance_differences.md)。']
        lines += ['', '这些数组含稀疏索引等结构数据，数量不等于 LR 对数。P 值与整数结构要求精确一致。',
                  'Visium、多空间样本、多组学三本原教程默认不保存对象，因此没有这项 pickle 补充检查；',
                  '它们的 LR/聚合数组、完整代码执行和图文件证据单独保留，不冒称导出了所有路径数组。',
                  '[补充审计](validation/tutorial_saved_object_comparison.json)。', '']
    lines += ['## 推断核性能', '',
              '时间为已完成独立进程重复的中位数。`倍数 = 原版核时间 / 新版核时间`。',
              'RSS 是推断阶段采样峰值，含该进程已加载的 Python 依赖和输入；不是只计算新分配数组。', '',
              '推断内存每 20 ms 采样，整个进程由外部每 100 ms 采样；后者可能漏掉短暂峰值，因此两列不是同一采样序列的包含关系。', '',
              '| 实验目录 | 数据 | 完成重复 原/新 | 原版核秒 | 新版核秒 | 倍数 | 原版 RSS MiB | 新版 RSS MiB | 数值对照 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    chart_rows = []
    process_rows = []
    limit_notes = []
    for path in sorted((ROOT / 'results/benchmarks').glob('*/summary.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        old = [r for r in data['records'] if r['engine'] == 'upstream']
        new = [r for r in data['records'] if r['engine'] == 'streaming']
        t0, t1 = median(old, 'inference_seconds'), median(new, 'inference_seconds')
        m0, m1 = median(old, 'inference_peak_rss_bytes'), median(new, 'inference_peak_rss_bytes')
        speedup = t0 / t1 if t0 is not None and t1 and data.get('all_comparisons_pass', False) and data.get('comparisons') else None
        n0, n1 = sum(r['status'] == 'completed' for r in old), sum(r['status'] == 'completed' for r in new)
        passed = data.get('all_comparisons_pass', False)
        comparison_status = ('通过' if passed else
                             '原版受资源限额，无配对' if n0 == 0 and n1 > 0 and
                             any(r.get('status') in {'timeout', 'memory_budget_exceeded', 'private_memory_budget_exceeded'} for r in old)
                             else '未完成或未通过，见记录')
        process_rows.append((path.parent.name, median(old, 'process_seconds'), median(new, 'process_seconds'),
                             median(old, 'process_peak_rss_bytes'), median(new, 'process_peak_rss_bytes')))
        for record in data['records']:
            if record.get('status') != 'completed':
                limit_notes.append(f"- `{path.parent.name}` / {record['engine']} / repeat={record.get('repeat')}: "
                                   f"`{record.get('status')}`，进程运行 {number(record.get('process_seconds'))} 秒；"
                                   f"限额 {data.get('limits')}。未完成配对，不计算加速倍数。")
        lines.append(f"| {path.parent.name} | {data['dataset']} | {n0}/{n1} | {number(t0)} | {number(t1)} | "
                     f"{number(speedup)} | {number(m0 / 2**20 if m0 else None)} | {number(m1 / 2**20 if m1 else None)} | "
                     f"{comparison_status} |")
        if speedup and passed:
            chart_rows.append((path.parent.name, t0, t1, m0 / 2**20, m1 / 2**20))
    lines += ['', '各目录保留每个进程的命令、线程数、核时间、整个进程时间、内存、输出数组和比较。',
              '最终兼容性修补后，开发版进程已重新执行；原版源码未变，保留本次此前完成的原版进程作对照。',
              '被替代的开发版记录保存在对应目录的 history，summary 的 refresh_history 记录更新原因与原版文件校验。',
              '只有一次配对的结果是本机的一次观测，不能视作多次重复的稳定估计或跨硬件保证。',
              '第一次数值调用可能包含 JIT 初始化或缓存读取。不是每次均重新编译；不能用该结果宣称无缓存冷启动性能。',
              '若原版因时间或内存上限退出，只报告该限制和新引擎完成结果，不虚构配对加速倍数或结果一致性。', '',
              '## 完整基准进程', '',
              '包括 Python 导入、此基准的数据加载/生成与对象准备、推断及结果保存；不含独立的数据下载/预处理，',
              '也不等于包含候选筛选、下游分析和绘图的完整生物学工作流。数值为已完成重复的中位数。', '',
              '| 实验目录 | 原版进程秒 | 新版进程秒 | 原版进程峰值 RSS MiB | 新版进程峰值 RSS MiB |',
              '|---|---:|---:|---:|---:|']
    for label, t0, t1, m0, m1 in process_rows:
        lines.append(f'| {label} | {number(t0)} | {number(t1)} | '
                     f'{number(m0 / 2**20 if m0 else None)} | {number(m1 / 2**20 if m1 else None)} |')
    if limit_notes:
        lines += ['', '未完成记录：', '', *limit_notes]
    lines += ['',
              '## 数据及边界', '',
              '- 真实 PBMC 数据来自 10x Genomics，下载 URL、文件大小及 SHA256 在 `data/pbmc68k_download_manifest.json`。',
              '- 68,579 个细胞均保留；原始表达含 32,738 个基因。归一化 library size 来自全基因计数，随后保留信号基因。',
              '- 官方旧分析只提供部分细胞标签。完整数据采用固定参数的 SVD/KMeans 计算分组；这是性能实验分组，不声称是经验证的细胞类型注释。',
              '- 性能实验固定相同候选 LR，未将候选筛选或 DE 纳入推断核时间；全部教程则按原工作流执行。',
              '- 合成 100k/1m 等规模为独立生成的计数，明确不是百万细胞真实图谱验证。',
              '- 合成十万/百万实验固定 256 个输入基因、10 个细胞群、100 对 LR、100 次置换；当前候选依赖闭包含 59 个统计基因。这不能代表任意群数或全部 LR 的百万细胞负载。',
              '- 本机物理内存约 15.71 GiB，其他程序占用较多。大型任务串行运行并监控 RSS/private memory；未测量换页 I/O，内存压力和后台活动可能影响墙钟时间。',
              '- 本次优化细胞群层面推断，不宣称优化 spot 层面推断、全部预处理或下游中心性算法。',
              '- 性能基线是固定的 CellChatPy GitHub 发布源码；本次没有测量 R CellChat 或其他 Python 实现，不能据此宣称超越这些软件。',
              '- 软件和统计定义一致性不是生物学真值验证，也不是方法首创性证明。', '',
              '设计说明见 [METHOD.md](METHOD.md)。审查见 `validation/code_review.md` 和 `validation/final_independent_review.md`。', '']
    ablation = ROOT / 'validation/ablation.json'
    if ablation.exists():
        data = json.loads(ablation.read_text(encoding='utf-8'))
        lines += ['## 消融实验', '',
                  '另有小规模、单进程预热后的消融，用于定位优化贡献；不能与上表的独立进程计时混用。',
                  '5,000 细胞、256 输入基因、100 对 LR、20 次置换，每个变体三次计时。', '',
                  '| 变体 | 中位秒 | 相对完整引擎耗时 | 统计基因数 | 概率及 P 值精确一致 |',
                  '|---|---:|---:|---:|---|']
        labels = {'all_enabled': '完整引擎', 'no_pruning': '关闭静态裁剪', 'no_sparse_plan': '关闭稀疏顺序统计复用'}
        for variant, row in data['results'].items():
            equal = all(row['exact_prob']) and all(row['exact_pval'])
            lines.append(f"| {labels.get(variant, variant)} | {number(row['median_seconds'], 4)} | "
                         f"{number(row['median_time_relative_to_all_enabled'])} | "
                         f"{row['evaluated_genes'][0]} | {'是' if equal else '否'} |")
        lines += ['', '本例支持复用稀疏顺序统计的时间收益。静态裁剪减少了统计基因数，',
                  '但关闭裁剪反而略快，因此本实验不能支持“裁剪普遍提速”的结论。',
                  '短任务计时、执行开销和机器后台负载都可能影响此小消融；最终重跑时未同时执行教程。',
                  '[原始消融记录](validation/ablation.json)。', '']
    if chart_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        out = ROOT / 'results/figures'
        out.mkdir(exist_ok=True)
        positions = np.arange(len(chart_rows))
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.5 * len(chart_rows)), 4.7))
        for ax, offsets, ylabel in [(axes[0], (1, 2), 'Inference time (s, log scale)'),
                                    (axes[1], (3, 4), 'Inference peak RSS (MiB)')]:
            ax.bar(positions - .19, [r[offsets[0]] for r in chart_rows], .38, label='Pinned upstream', color='#64748b')
            ax.bar(positions + .19, [r[offsets[1]] for r in chart_rows], .38, label='Streaming', color='#0d9488')
            ax.set_xticks(positions, [r[0] for r in chart_rows], rotation=20, ha='right', fontsize=8)
            ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False)
        axes[0].set_yscale('log')
        fig.suptitle('Completed paired benchmarks — medians; inference kernel only')
        fig.tight_layout()
        fig.savefig(out / 'benchmark_summary.png', dpi=180, bbox_inches='tight')
        fig.savefig(out / 'benchmark_summary.svg', bbox_inches='tight')
        plt.close(fig)
        lines += ['![推断核性能](results/figures/benchmark_summary.png)', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(ROOT / 'RESULTS.md')


if __name__ == '__main__':
    main()
