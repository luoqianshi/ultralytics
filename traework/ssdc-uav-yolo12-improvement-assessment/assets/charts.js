(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var catLoss = style.getPropertyValue('--cat-loss').trim();
  var catStruct = style.getPropertyValue('--cat-struct').trim();
  var catCombo = style.getPropertyValue('--cat-combo').trim();
  var catBase = style.getPropertyValue('--cat-base').trim();
  var catOther = style.getPropertyValue('--cat-other').trim();

  var CAT_COLOR = { loss: catLoss, struct: catStruct, combo: catCombo, base: catBase, other: catOther };

  // 数据：SSDC-UAV 测试集（s 尺度、COCO 预训练、150 epoch，除标注外）
  // [名称, 类别, P, R, F1, mAP50, mAP75, mAP50-95, 参数量M, GFLOPS]
  var S150 = [
    ['YOLO12s 基线', 'base', 84.581, 82.269, 83.409, 88.425, 61.023, 55.814, 9.25, 21.52],
    ['Focaler-CIoU', 'loss', 85.047, 81.808, 83.396, 88.656, 61.142, 55.792, 9.25, 21.52],
    ['SD-Loss', 'loss', 84.693, 81.719, 83.179, 88.455, 60.839, 55.821, 9.25, 21.52],
    ['PIoU2', 'loss', 85.573, 81.262, 83.362, 88.539, 61.550, 55.953, 9.25, 21.52],
    ['DySample', 'struct', 85.162, 81.515, 83.299, 88.371, 61.247, 55.823, 9.28, 21.56],
    ['P2 头', 'struct', 85.112, 80.760, 82.879, 88.006, 59.884, 55.052, 9.38, 28.36],
    ['P2+DySample', 'struct', 84.765, 80.778, 82.724, 87.931, 60.163, 55.098, 9.41, 28.44],
    ['SPDConv', 'struct', 81.058, 75.496, 78.178, 81.050, 43.935, 44.608, 9.31, 0],
    ['MoCA', 'struct', 82.357, 79.215, 80.755, 86.587, 58.811, 54.009, 6.55, 16.45],
    ['A2C2f-Mona', 'struct', 85.197, 79.988, 82.510, 88.153, 60.938, 55.588, 6.81, 18.44],
    ['A2C2f-SCSA+CBAM', 'struct', 82.357, 79.899, 81.110, 86.848, 58.410, 53.991, 6.41, 16.32],
    ['A2C2f-EMA', 'struct', 84.721, 80.955, 82.795, 88.012, 60.310, 55.385, 6.87, 18.55],
    ['EMA（颈部）', 'struct', 84.018, 81.479, 82.729, 88.028, 59.518, 54.892, 9.26, 21.60],
    ['Mona（颈部）', 'struct', 83.876, 79.100, 81.418, 87.152, 58.994, 54.367, 5.66, 14.08],
    ['MultiScaleGatedAttn', 'struct', 84.932, 81.575, 83.220, 88.403, 60.926, 55.547, 13.46, 29.81],
    ['Focaler-CIoU+DySample', 'combo', 85.342, 80.795, 83.006, 88.279, 60.572, 55.558, 9.28, 21.56],
    ['PIoU2+DySample', 'combo', 85.397, 80.849, 83.061, 88.491, 61.620, 55.865, 9.28, 21.56],
    ['YOLOv11s', 'other', 84.997, 80.503, 82.689, 87.916, 59.517, 54.968, 9.43, 21.55],
    ['YOLOv8s', 'other', 84.438, 80.792, 82.575, 87.665, 59.866, 55.013, 11.14, 28.65],
    ['YOLOv5su', 'other', 84.887, 79.993, 82.368, 87.889, 60.344, 55.253, 9.12, 24.04],
    ['YOLOv26s', 'other', 83.292, 80.050, 81.639, 86.607, 59.011, 53.971, 9.95, 22.50]
  ];

  function init(id, opts) {
    var el = document.getElementById(id);
    if (!el) return null;
    var c = echarts.init(el, null, { renderer: 'svg' });
    c.setOption(opts);
    window.addEventListener('resize', function () { c.resize(); });
    return c;
  }

  var axisText = { color: muted, fontSize: 11 };

  // ---------- 图 1：mAP50 横向对比 ----------
  var sorted = S150.slice().sort(function (a, b) { return a[5] - b[5]; });
  init('chart-map50', {
    animation: false,
    grid: { left: 190, right: 60, top: 20, bottom: 36 },
    tooltip: {
      appendToBody: true,
      formatter: function (p) {
        var d = p.data.meta;
        return '<b>' + d[0] + '</b><br/>mAP50: ' + d[5].toFixed(3) + '<br/>P: ' + d[2].toFixed(3) +
          ' / R: ' + d[3].toFixed(3) + '<br/>参数量: ' + d[8] + 'M';
      }
    },
    xAxis: {
      type: 'value', min: 79, max: 89.6,
      axisLabel: axisText, splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'category',
      data: sorted.map(function (d) { return d[0]; }),
      axisLabel: { color: ink, fontSize: 11 }, axisTick: { show: false }
    },
    series: [{
      type: 'bar',
      data: sorted.map(function (d) {
        return { value: d[5], itemStyle: { color: CAT_COLOR[d[1]] }, meta: d };
      }),
      barWidth: 13,
      label: { show: true, position: 'right', fontSize: 10, color: ink, formatter: function (p) { return p.value.toFixed(3); } },
      markLine: {
        symbol: 'none', silent: true,
        lineStyle: { color: catBase, type: 'dashed', width: 1 },
        label: { formatter: '基线 88.425', position: 'insideEndTop', fontSize: 10, color: catBase },
        data: [{ xAxis: 88.425 }]
      }
    }]
  });

  // ---------- 图 2：Precision-Recall 散点（气泡=参数量） ----------
  init('chart-pr', {
    animation: false,
    grid: { left: 60, right: 40, top: 30, bottom: 50 },
    tooltip: {
      appendToBody: true,
      formatter: function (p) {
        var d = p.data.meta;
        return '<b>' + d[0] + '</b><br/>P: ' + d[2].toFixed(3) + ' / R: ' + d[3].toFixed(3) +
          '<br/>F1: ' + d[4].toFixed(3) + ' / mAP50: ' + d[5].toFixed(3) + '<br/>参数量: ' + d[8] + 'M';
      }
    },
    xAxis: {
      type: 'value', name: 'Recall (%)', nameTextStyle: { color: muted }, min: 78.5, max: 82.8,
      axisLabel: axisText, splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'value', name: 'Precision (%)', nameTextStyle: { color: muted }, min: 80.3, max: 86.2,
      axisLabel: axisText, splitLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'scatter',
      data: S150.map(function (d) {
        return {
          value: [d[3], d[2]],
          symbolSize: Math.sqrt(d[8]) * 5.2,
          itemStyle: { color: CAT_COLOR[d[1]], opacity: 0.85 },
          label: { show: true, position: 'top', fontSize: 9.5, color: ink, formatter: d[0] },
          meta: d
        };
      }),
      labelLayout: { hideOverlap: true },
      markLine: {
        symbol: 'none', silent: true,
        lineStyle: { color: catBase, type: 'dashed', width: 1 },
        label: { fontSize: 9.5, color: catBase },
        data: [
          { xAxis: 82.269, label: { formatter: '基线 R 82.269', position: 'insideEndTop' } },
          { yAxis: 84.581, label: { formatter: '基线 P 84.581', position: 'insideEndBottom' } }
        ]
      },
      markArea: {
        silent: true, itemStyle: { color: accent, opacity: 0.05 },
        data: [[{ xAxis: 82.269, yAxis: 84.581 }, { xAxis: 82.8, yAxis: 86.2 }]]
      }
    }]
  });

  // ---------- 图 3：关键改进相对基线的 Δ 指标 ----------
  var deltaCats = ['Focaler-CIoU', 'SD-Loss', 'PIoU2', 'DySample', 'Focaler+Dy', 'PIoU2+Dy'];
  var deltaVals = {
    'ΔPrecision': [0.466, 0.112, 0.992, 0.581, 0.761, 0.816],
    'ΔRecall': [-0.461, -0.550, -1.007, -0.754, -1.474, -1.420],
    'ΔF1': [-0.013, -0.230, -0.047, -0.110, -0.403, -0.348],
    'ΔmAP50': [0.231, 0.030, 0.114, -0.054, -0.146, 0.066]
  };
  init('chart-delta', {
    animation: false,
    grid: { left: 130, right: 50, top: 46, bottom: 36 },
    legend: { top: 0, textStyle: { color: muted, fontSize: 11 } },
    tooltip: { appendToBody: true, valueFormatter: function (v) { return (v >= 0 ? '+' : '') + v.toFixed(3); } },
    xAxis: {
      type: 'value', min: -1.7, max: 1.3,
      axisLabel: axisText, splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'category', data: deltaCats,
      axisLabel: { color: ink, fontSize: 11 }, axisTick: { show: false }
    },
    series: Object.keys(deltaVals).map(function (k) {
      return {
        name: k, type: 'bar', barWidth: 9, data: deltaVals[k],
        itemStyle: { color: function (p) { return p.value >= 0 ? accent : accent2; } }
      };
    })
  });

  // ---------- 图 4：参数量-mAP50 气泡 ----------
  init('chart-params', {
    animation: false,
    grid: { left: 60, right: 40, top: 30, bottom: 50 },
    tooltip: {
      appendToBody: true,
      formatter: function (p) {
        var d = p.data.meta;
        return '<b>' + d[0] + '</b><br/>参数量: ' + d[8] + 'M / GFLOPs: ' + d[9] + '<br/>mAP50: ' + d[5].toFixed(3);
      }
    },
    xAxis: {
      type: 'value', name: '参数量 (M)', nameTextStyle: { color: muted }, min: 5, max: 14.2,
      axisLabel: axisText, splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'value', name: 'mAP50 (%)', nameTextStyle: { color: muted }, min: 80, max: 89.6,
      axisLabel: axisText, splitLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'scatter',
      data: S150.map(function (d) {
        return {
          value: [d[8], d[5]], symbolSize: 12,
          itemStyle: { color: CAT_COLOR[d[1]], opacity: 0.85 },
          label: { show: true, position: 'top', fontSize: 9.5, color: ink, formatter: d[0] },
          meta: d
        };
      }),
      labelLayout: { hideOverlap: true },
      markLine: {
        symbol: 'none', silent: true,
        lineStyle: { color: catBase, type: 'dashed', width: 1 },
        label: { fontSize: 9.5, color: catBase, formatter: '基线 9.25M' },
        data: [{ xAxis: 9.25 }]
      },
      markArea: {
        silent: true, itemStyle: { color: accent, opacity: 0.05 },
        data: [[{ xAxis: 8.84 }, { xAxis: 9.77 }]]
      }
    }]
  });

  // ---------- Mermaid ----------
  if (window.mermaid) {
    mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
  }
})();
