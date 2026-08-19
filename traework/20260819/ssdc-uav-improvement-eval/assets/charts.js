(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  function font(size, weight) {
    return weight + ' ' + size + "px 'InstrumentSans','PingFang SC','Microsoft YaHei',sans-serif";
  }

  window.addEventListener('resize', function () {
    charts.forEach(function (c) { c.resize(); });
  });
  var charts = [];

  // --- Chart 1: 各改进方案相对重跑基线的增量 ---
  var el1 = document.getElementById('chart-delta');
  if (el1) {
    var c1 = echarts.init(el1, null, { renderer: 'svg' });
    charts.push(c1);
    var schemes = ['Focaler-CIoU', 'PIoU2', 'PIoU2-DySample', 'SD-Loss*', 'DySample-plus', 'DySample', 'MSGA', 'WIoUv3*', 'Focaler-DySample', 'DySample-WIoUv3', 'A2C2f_Mona', 'EMA', 'P2', 'P2-DySample'];
    var d50 = [0.389, 0.272, 0.224, 0.188, 0.175, 0.104, 0.136, 0.166, 0.012, -0.012, -0.114, -0.239, -0.261, -0.336];
    var d5095 = [0.153, 0.314, 0.226, 0.182, 0.228, 0.184, -0.092, 0.034, -0.081, -0.100, -0.051, -0.747, -0.587, -0.541];
    c1.setOption({
      animation: false,
      tooltip: { appendToBody: true, trigger: 'item' },
      legend: { top: 0, textStyle: { color: muted, fontSize: 12 } },
      grid: { left: 130, right: 30, top: 34, bottom: 28 },
      xAxis: {
        type: 'value', name: 'Δ (%)',
        nameTextStyle: { color: muted, fontSize: 11 },
        axisLabel: { color: muted, fontSize: 11, formatter: function (v) { return v > 0 ? '+' + v : v; } },
        splitLine: { lineStyle: { color: rule } }
      },
      yAxis: {
        type: 'category', inverse: true, data: schemes,
        axisLabel: { color: ink, fontSize: 11.5, fontFamily: font(11.5, 400) },
        axisLine: { lineStyle: { color: rule } }, axisTick: { show: false }
      },
      series: [
        {
          name: 'ΔmAP50', type: 'bar', data: d50, barWidth: 8,
          itemStyle: { color: function (p) { return p.value >= 0 ? accent : accent2; }, borderRadius: 2 },
          label: { show: true, position: 'right', fontSize: 10, color: muted, formatter: function (p) { return (p.value >= 0 ? '+' : '') + p.value.toFixed(2); } }
        },
        {
          name: 'ΔmAP50-95', type: 'bar', data: d5095, barWidth: 8,
          itemStyle: { color: function (p) { return p.value >= 0 ? accent : accent2; }, opacity: 0.45, borderRadius: 2 },
          label: { show: true, position: 'right', fontSize: 10, color: muted, formatter: function (p) { return (p.value >= 0 ? '+' : '') + p.value.toFixed(2); } }
        }
      ]
    });
  }

  // --- Chart 2: 噪声地板标定 ---
  var el2 = document.getElementById('chart-noise');
  if (el2) {
    var c2 = echarts.init(el2, null, { renderer: 'svg' });
    charts.push(c2);
    var names = ['新基线(8/19)', 'DySample', 'Focaler-DySample', 'MSGA', '旧基线(auto)', 'DySample-plus', 'SD-Loss*', 'PIoU2-DySample', 'WIoUv3*', 'PIoU2', 'Focaler-CIoU'];
    var vals = [88.267, 88.371, 88.279, 88.403, 88.425, 88.442, 88.455, 88.491, 88.433, 88.539, 88.656];
    var isCiou = [true, false, false, false, true, false, true, false, false, false, false];
    c2.setOption({
      animation: false,
      tooltip: { appendToBody: true, trigger: 'item' },
      grid: { left: 130, right: 46, top: 26, bottom: 30 },
      xAxis: {
        type: 'value', min: 88.1, max: 88.75,
        axisLabel: { color: muted, fontSize: 11 },
        splitLine: { lineStyle: { color: rule } }
      },
      yAxis: {
        type: 'category', inverse: true, data: names,
        axisLabel: { color: ink, fontSize: 11.5 },
        axisLine: { lineStyle: { color: rule } }, axisTick: { show: false }
      },
      series: [{
        type: 'bar', barWidth: 14, data: vals.map(function (v, i) {
          return { value: v, itemStyle: { color: isCiou[i] ? accent2 : accent, opacity: isCiou[i] ? 0.75 : 0.9 } };
        }),
        label: { show: true, position: 'right', fontSize: 10.5, color: muted, formatter: function (p) { return p.value.toFixed(3); } },
        markLine: {
          symbol: 'none', silent: true,
          data: [
            { xAxis: 88.382, label: { formatter: 'CIoU 等效均值 88.382', color: ink, fontSize: 10.5, position: 'insideEndTop' }, lineStyle: { color: ink, type: 'solid', width: 1.5 } },
            { xAxis: 88.182, label: { formatter: '−2σ', color: muted, fontSize: 10, position: 'insideEndTop' }, lineStyle: { color: muted, type: 'dashed' } },
            { xAxis: 88.582, label: { formatter: '+2σ', color: muted, fontSize: 10, position: 'insideEndTop' }, lineStyle: { color: muted, type: 'dashed' } }
          ]
        }
      }]
    });
  }

  // --- Chart 3: P/R 协同散点 ---
  var el3 = document.getElementById('chart-pr');
  if (el3) {
    var c3 = echarts.init(el3, null, { renderer: 'svg' });
    charts.push(c3);
    var pts = [
      { name: '基线(重跑)', p: 84.911, r: 81.341, g: 'c' },
      { name: '旧基线(auto)', p: 84.581, r: 82.269, g: 'c' },
      { name: 'SD-Loss*', p: 84.693, r: 81.719, g: 'c' },
      { name: 'Focaler-CIoU', p: 85.047, r: 81.808, g: 'a' },
      { name: 'PIoU2', p: 85.573, r: 81.262, g: 'a' },
      { name: 'PIoU2-DySample', p: 85.397, r: 80.849, g: 'a' },
      { name: 'DySample-plus', p: 85.511, r: 81.170, g: 'a' },
      { name: 'P2', p: 85.112, r: 80.760, g: 'n' },
      { name: 'EMA', p: 84.018, r: 81.479, g: 'n' },
      { name: 'A2C2f_Mona', p: 85.197, r: 79.988, g: 'n' }
    ];
    var colorMap = { c: accent2, a: accent, n: muted };
    c3.setOption({
      animation: false,
      tooltip: {
        appendToBody: true,
        formatter: function (p) { return p.data.name + '<br/>P: ' + p.data.p + ' · R: ' + p.data.r; }
      },
      grid: { left: 52, right: 110, top: 26, bottom: 44 },
      xAxis: {
        type: 'value', name: 'Precision (%)', min: 83.8, max: 85.8,
        nameTextStyle: { color: muted }, axisLabel: { color: muted, fontSize: 11 }, splitLine: { lineStyle: { color: rule } }
      },
      yAxis: {
        type: 'value', name: 'Recall (%)', min: 79.8, max: 82.5,
        nameTextStyle: { color: muted }, axisLabel: { color: muted, fontSize: 11 }, splitLine: { lineStyle: { color: rule } }
      },
      series: [{
        type: 'scatter',
        symbolSize: 14,
        data: pts.map(function (d) { return { name: d.name, value: [d.p, d.r], p: d.p, r: d.r, itemStyle: { color: colorMap[d.g], opacity: 0.9 } }; }),
        label: {
          show: true, position: 'right', distance: 8, fontSize: 10.5, color: ink,
          formatter: function (p) { return p.data.name; }
        },
        markLine: {
          symbol: 'none', silent: true,
          lineStyle: { color: rule, type: 'dashed' },
          data: [{ xAxis: 84.728 }, { yAxis: 81.776 }]
        }
      }]
    });
  }

  // --- Chart 4: 数据集目标尺寸分布 ---
  var el4 = document.getElementById('chart-dataset');
  if (el4) {
    var c4 = echarts.init(el4, null, { renderer: 'svg' });
    charts.push(c4);
    c4.setOption({
      animation: false,
      tooltip: { appendToBody: true },
      legend: { top: 0, textStyle: { color: muted, fontSize: 12 } },
      grid: { left: 48, right: 30, top: 34, bottom: 34 },
      xAxis: {
        type: 'category', data: ['小目标\n(<0.02)', '中目标\n(0.02–0.05)', '大目标\n(≥0.05)'],
        axisLabel: { color: ink, fontSize: 11.5 }, axisLine: { lineStyle: { color: rule } }, axisTick: { show: false }
      },
      yAxis: { type: 'value', axisLabel: { color: muted, fontSize: 11, formatter: '{value}%' }, splitLine: { lineStyle: { color: rule } } },
      series: [
        { name: '按宽度 w 统计', type: 'bar', barWidth: 34, data: [0.24, 10.33, 89.44], itemStyle: { color: accent, borderRadius: 3 } },
        { name: '按高度 h 统计', type: 'bar', barWidth: 34, data: [0.21, 12.46, 87.33], itemStyle: { color: accent2, opacity: 0.65, borderRadius: 3 } }
      ],
      label: { show: true, position: 'top', fontSize: 10.5, color: muted, formatter: '{c}%' }
    });
  }

  if (window.mermaid) {
    mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
  }
})();
