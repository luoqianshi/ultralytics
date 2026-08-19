(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var neg = style.getPropertyValue('--neg').trim();
  var warn = style.getPropertyValue('--warn').trim();

  var axisCommon = {
    axisLine: { lineStyle: { color: rule } },
    axisLabel: { color: muted, fontSize: 11 },
    splitLine: { lineStyle: { color: rule, type: 'dashed' } }
  };

  // --- Chart 1: ΔmAP50-95 全方案（vs 150e 基线，发散条形图） ---
  var el1 = document.getElementById('chart-delta');
  if (el1) {
    var data1 = [
      ['A2C2f_SCSA_CBAM', -1.823], ['MoCA', -1.805], ['Mona(整块)', -1.447],
      ['A2C2f_FCM(300e)', -0.917], ['EMA(附加)', -0.922], ['P2', -0.762],
      ['P2-DySample', -0.716], ['A2C2f_EMA', -0.429], ['A2C2f_MCA(300e)', -0.335],
      ['MultiScaleGatedAttn', -0.267], ['DySample-WIoUv3', -0.275], ['Focaler-CIoU-DySample', -0.256],
      ['A2C2f_Mona', -0.226], ['WIoUv3(194e)', -0.141], ['Focaler-CIoU', -0.022],
      ['SD-Loss', 0.007], ['DySample', 0.009], ['PIoU2-DySample', 0.051], ['PIoU2', 0.139]
    ];
    var c1 = echarts.init(el1, null, { renderer: 'svg' });
    c1.setOption({
      animation: false,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(3); } },
      grid: { left: 190, right: 40, top: 10, bottom: 30 },
      xAxis: Object.assign({ type: 'value', name: 'ΔmAP50-95 (pp)' }, axisCommon),
      yAxis: { type: 'category', data: data1.map(function (d) { return d[0]; }), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: ink, fontSize: 11 } },
      series: [{
        type: 'bar', barWidth: 13,
        data: data1.map(function (d) { return { value: d[1], itemStyle: { color: d[1] >= 0 ? accent : neg, borderRadius: d[1] >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3] } }; }),
        label: { show: true, position: 'right', color: muted, fontSize: 10, formatter: function (p) { return (p.value >= 0 ? '+' : '') + p.value.toFixed(3); } }
      }]
    });
    window.addEventListener('resize', function () { c1.resize(); });
  }

  // --- Chart 2: ΔP vs ΔR 散点 ---
  var el2 = document.getElementById('chart-pr');
  if (el2) {
    var pr = [
      { name: 'PIoU2', p: 0.992, r: -1.007, g: 'loss' },
      { name: 'Focaler-CIoU', p: 0.466, r: -0.461, g: 'loss' },
      { name: 'SD-Loss', p: 0.112, r: -0.550, g: 'loss' },
      { name: 'WIoUv3', p: 0.537, r: -0.772, g: 'loss' },
      { name: 'DySample', p: 0.581, r: -0.754, g: 'struct' },
      { name: 'PIoU2-DySample', p: 0.816, r: -1.420, g: 'combo' },
      { name: 'Focaler-CIoU-DySample', p: 0.761, r: -1.474, g: 'combo' },
      { name: 'DySample-WIoUv3', p: 0.112, r: -0.994, g: 'combo' },
      { name: 'P2', p: 0.531, r: -1.509, g: 'struct' },
      { name: 'P2-DySample', p: 0.184, r: -1.491, g: 'combo' },
      { name: 'A2C2f_Mona', p: 0.616, r: -2.281, g: 'struct' },
      { name: 'A2C2f_EMA', p: 0.140, r: -1.314, g: 'struct' },
      { name: 'EMA(附加)', p: -0.563, r: -0.790, g: 'struct' }
    ];
    var groups = { loss: accent, struct: accent2, combo: warn };
    var c2 = echarts.init(el2, null, { renderer: 'svg' });
    c2.setOption({
      animation: false,
      tooltip: { trigger: 'item', appendToBody: true, formatter: function (p) { return p.data.name + '<br>ΔP: ' + p.data.p.toFixed(3) + '<br>ΔR: ' + p.data.r.toFixed(3); } },
      legend: { data: ['损失类', '结构类', '组合类'], textStyle: { color: muted, fontSize: 11 }, top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 45 },
      xAxis: Object.assign({ type: 'value', name: 'ΔPrecision (pp)', min: -1, max: 1.2 }, axisCommon),
      yAxis: Object.assign({ type: 'value', name: 'ΔRecall (pp)', min: -2.6, max: 0.4 }, axisCommon),
      series: [
        { name: '损失类', type: 'scatter', symbolSize: 11, itemStyle: { color: groups.loss, opacity: 0.9 }, data: pr.filter(function (d) { return d.g === 'loss'; }).map(function (d) { return { name: d.name, value: [d.p, d.r], p: d.p, r: d.r }; }), label: { show: true, position: 'top', color: ink, fontSize: 10, formatter: '{b}' } },
        { name: '结构类', type: 'scatter', symbolSize: 11, itemStyle: { color: groups.struct, opacity: 0.9 }, data: pr.filter(function (d) { return d.g === 'struct'; }).map(function (d) { return { name: d.name, value: [d.p, d.r], p: d.p, r: d.r }; }), label: { show: true, position: 'top', color: ink, fontSize: 10, formatter: '{b}' } },
        { name: '组合类', type: 'scatter', symbolSize: 11, itemStyle: { color: groups.combo, opacity: 0.9 }, data: pr.filter(function (d) { return d.g === 'combo'; }).map(function (d) { return { name: d.name, value: [d.p, d.r], p: d.p, r: d.r }; }), label: { show: true, position: 'top', color: ink, fontSize: 10, formatter: '{b}' } }
      ]
    });
    window.addEventListener('resize', function () { c2.resize(); });
  }

  // --- Chart 3: 关键方案 × 双基线对照 ---
  var el3 = document.getElementById('chart-dual-base');
  if (el3) {
    var schemes = ['Focaler-CIoU', 'SD-Loss', 'PIoU2', 'DySample', 'PIoU2-DySample'];
    var vs150 = [0.231, 0.030, 0.114, -0.054, 0.066];
    var vsSGD = [0.337, 0.140, 0.272, 0.142, 0.184];
    var c3 = echarts.init(el3, null, { renderer: 'svg' });
    c3.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return (v >= 0 ? '+' : '') + Number(v).toFixed(3); } },
      legend: { data: ['vs 150e基线(optimizer=auto)', 'vs 191e基线(SGD,严格对照)'], textStyle: { color: muted, fontSize: 11 }, top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: Object.assign({ type: 'category', data: schemes }, axisCommon, { axisLabel: { color: ink, fontSize: 11 }, splitLine: { show: false } }),
      yAxis: Object.assign({ type: 'value', name: 'ΔmAP50-95 (pp)' }, axisCommon),
      series: [
        { name: 'vs 150e基线(optimizer=auto)', type: 'bar', barWidth: 22, itemStyle: { color: muted, opacity: 0.55, borderRadius: [3, 3, 0, 0] }, data: vs150 },
        { name: 'vs 191e基线(SGD,严格对照)', type: 'bar', barWidth: 22, itemStyle: { color: accent, borderRadius: [3, 3, 0, 0] }, data: vsSGD, label: { show: true, position: 'top', color: ink, fontSize: 10, formatter: function (p) { return '+' + p.value.toFixed(3); } } }
      ]
    });
    window.addEventListener('resize', function () { c3.resize(); });
  }

  // --- Chart 4: 数据集尺寸分布（按飞行高度，官方全量统计） ---
  var el4 = document.getElementById('chart-dataset');
  if (el4) {
    var c4 = echarts.init(el4, null, { renderer: 'svg' });
    c4.setOption({
      animation: false,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(1) + '%'; } },
      legend: { data: ['小目标(<32²px)', '中目标(32²~96²px)', '大目标(>96²px)'], textStyle: { color: muted, fontSize: 11 }, top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: Object.assign({ type: 'category', data: ['5m 高度\n(均107×99px)', '8m 高度\n(均89×81px)', '10m 高度\n(均71×65px)', '全量 train\n(均85.7×78.5px)'] }, axisCommon, { axisLabel: { color: ink, fontSize: 11 }, splitLine: { show: false } }),
      yAxis: Object.assign({ type: 'value', max: 100, axisLabel: { color: muted, fontSize: 11, formatter: '{value}%' }, splitLine: { lineStyle: { color: rule, type: 'dashed' } }, axisLine: { lineStyle: { color: rule } } }),
      series: [
        { name: '小目标(<32²px)', type: 'bar', stack: 's', barWidth: 46, itemStyle: { color: accent2 }, data: [4.4, 8.4, 13.3, 8.9] },
        { name: '中目标(32²~96²px)', type: 'bar', stack: 's', itemStyle: { color: accent }, data: [50.5, 58.6, 69.9, 62.3] },
        { name: '大目标(>96²px)', type: 'bar', stack: 's', itemStyle: { color: muted, opacity: 0.55 }, data: [45.1, 33.0, 16.8, 28.8], label: { show: true, position: 'inside', color: '#fff', fontSize: 10, formatter: function (p) { return p.value.toFixed(1); } } }
      ]
    });
    window.addEventListener('resize', function () { c4.resize(); });
  }

  // --- Chart 5: GFLOPs vs mAP50-95 效率散点 ---
  var el5 = document.getElementById('chart-efficiency');
  if (el5) {
    var eff = [
      { n: 'YOLOv12s 基线', g: 21.52, m: 55.814, p: 9.25, c: neg, keep: true },
      { n: 'PIoU2', g: 21.52, m: 55.953, p: 9.25, c: accent, keep: true },
      { n: 'PIoU2-DySample', g: 21.56, m: 55.865, p: 9.28, c: accent, keep: true },
      { n: 'DySample', g: 21.56, m: 55.823, p: 9.28, c: accent, keep: true },
      { n: 'EMA(附加)', g: 21.60, m: 54.892, p: 9.26, c: accent2 },
      { n: 'P2', g: 28.36, m: 55.052, p: 9.38, c: accent2 },
      { n: 'P2-DySample', g: 28.44, m: 55.098, p: 9.41, c: accent2 },
      { n: 'MultiScaleGatedAttn', g: 29.81, m: 55.547, p: 13.46, c: accent2 },
      { n: 'A2C2f_Mona', g: 18.44, m: 55.588, p: 6.81, c: warn },
      { n: 'A2C2f_EMA', g: 18.55, m: 55.385, p: 6.87, c: warn },
      { n: 'A2C2f_MCA', g: 18.41, m: 55.479, p: 6.86, c: warn },
      { n: 'A2C2f_FCM', g: 18.39, m: 54.897, p: 7.49, c: warn },
      { n: 'MoCA', g: 16.45, m: 54.009, p: 6.55, c: warn },
      { n: 'Mona(整块)', g: 14.08, m: 54.367, p: 5.66, c: warn },
      { n: 'YOLOv12n 基线', g: 6.48, m: 54.735, p: 2.57, c: muted }
    ];
    var c5 = echarts.init(el5, null, { renderer: 'svg' });
    c5.setOption({
      animation: false,
      tooltip: { trigger: 'item', appendToBody: true, formatter: function (p) { return p.data.n + '<br>GFLOPs: ' + p.data.g + '<br>mAP50-95: ' + p.data.m + '<br>Params: ' + p.data.p + 'M'; } },
      grid: { left: 55, right: 30, top: 20, bottom: 45 },
      xAxis: Object.assign({ type: 'value', name: 'GFLOPs (640×640)', min: 5, max: 32 }, axisCommon),
      yAxis: Object.assign({ type: 'value', name: 'mAP50-95 (%)', min: 53.6, max: 56.3 }, axisCommon),
      series: [{
        type: 'scatter', symbolSize: 14,
        data: eff.map(function (d) { return { name: d.n, value: [d.g, d.m], n: d.n, g: d.g, m: d.m, p: d.p, itemStyle: { color: d.c } }; }),
        label: { show: true, position: 'top', color: ink, fontSize: 10, formatter: '{b}' },
        markLine: { silent: true, symbol: 'none', lineStyle: { color: neg, type: 'dashed' }, data: [{ yAxis: 55.814, label: { formatter: '基线 55.814', color: neg, fontSize: 10 } }] }
      }]
    });
    window.addEventListener('resize', function () { c5.resize(); });
  }
})();
