(function () {
  "use strict";
  const STORAGE_KEY = "wia_public_language";
  const DEFAULT_LANG = "zh";
  const MESSAGES = {
  "zh": {
    "common.switch_language": "切换语言",
    "common.switch_to_zh": "中文",
    "common.switch_to_en": "EN",
    "common.on": "开",
    "common.off": "关",
    "nav.home": "首页",
    "nav.capabilities": "能力",
    "nav.resources": "资源",
    "nav.data": "数据",
    "nav.code": "代码",
    "nav.paper": "论文",
    "nav.method": "方法",
    "nav.results": "结果",
    "index.title": "WIA · 世界智能体",
    "index.meta_description": "WIA 是面向数字智能体的基础世界模型，统一七类数字环境，连接预测与行动。",
    "index.hero.eyebrow": "面向数字智能体的基础世界模型",
    "index.hero.title": "世界<br /><span class=\"grad\">智能体</span>",
    "index.hero.subtitle": "让数字智能体先理解环境，再采取行动。WIA 统一七类数字世界，以基础世界模型驱动状态预测、模拟训练与主动决策。",
    "index.stat.cpt_samples": "CPT 交互样本",
    "index.stat.sft_trajectories": "SFT 轨迹",
    "index.stat.rl_samples": "RL 样本",
    "index.stat.digital_domains": "数字环境域",
    "index.features.kicker": "核心构建模块",
    "index.features.title": "从世界理解到世界行动",
    "index.features.subtitle": "WIA 不是被动模拟器，而是把环境预测能力延伸到智能体策略优化的完整体系。",
    "index.feature.corpus.title": "WIA-Corpus",
    "index.feature.corpus.desc": "覆盖 CPT、SFT、RL 的全生命周期训练语料，把异构数字交互统一成 action–observation 轨迹。",
    "index.feature.bench.title": "WIA-Bench",
    "index.feature.bench.desc": "首个成对 ID / OOD 世界模型评测集，1.8k 域内与 1.7k 域外样本，横跨七大数字环境。",
    "index.feature.wm.title": "WIA-WM",
    "index.feature.wm.desc": "4B / 9B / 27B / 35B-A3B / 122B-A10B 基础世界模型，系统化建模状态转移，并支持模拟 SFT、模拟 RL 与推理前瞻。",
    "index.feature.wam.title": "WIA-WAM",
    "index.feature.wam.desc": "通过 Dual-Perspective Inversion 与 CSPO 耦合预测和执行，让策略隐式内化环境动力学。",
    "index.feature.generalization.title": "跨域泛化",
    "index.feature.generalization.desc": "Android、MCP、OS、Search、SWE、Terminal、Web 七域统一建模，知识不再被单一界面割裂。",
    "index.feature.scaling.title": "规模化路线",
    "index.feature.scaling.desc": "从小模型到 122B-A10B MoE 大模型，验证世界模型能力随规模持续提升，支持 agent 训练与推理应用。",
    "index.matrix.kicker": "模型家族",
    "index.matrix.title": "世界模型矩阵",
    "index.matrix.subtitle": "统一模型家族覆盖从高效小模型到大规模 MoE。",
    "index.matrix.4b.desc": "高效世界模型基线",
    "index.matrix.9b.desc": "均衡推理与能力",
    "index.matrix.27b.desc": "Dense 高性能档位",
    "index.matrix.35b.desc": "MoE 高效扩展",
    "index.matrix.122b.desc": "大规模 MoE",
    "index.method.kicker": "流程",
    "index.method.title": "数据基座 · 世界推演 · 动作执行",
    "index.method.subtitle": "WIA 覆盖数据基座、世界推演与动作执行全链路：WIA-Corpus 与 WIA-Bench 建立数据与评测基础，WIA-FWM 建模状态转移与奖励动态，WIA-WAM 对齐动作生成与预测动态。",
    "index.method.unify.title": "统一环境",
    "index.method.unify.desc": "把七类数字界面序列化为共享的文本化 action–observation 表示。",
    "index.method.train.title": "训练 WIA-WM",
    "index.method.train.desc": "CPT → SFT → RL 三阶段训练，建立系统化下一观测预测能力。",
    "index.method.sim.title": "模拟扩展",
    "index.method.sim.desc": "以世界模型生成模拟数据、提供模拟奖励与推理前瞻。",
    "index.method.act.title": "耦合行动",
    "index.method.act.desc": "DPI 对齐双视角表示，CSPO 让行动选择与模拟后果直接关联。",
    "index.domains.kicker": "七大领域",
    "index.domains.title": "一个模型，覆盖数字智能体的主要战场",
    "index.results.kicker": "评测结果",
    "index.results.title": "用统一的基准衡量世界理解",
    "index.results.subtitle": "WIA-WM 在 WIA-Bench 上取得领先平均世界模型分数，并在 OOD 套件上显著超越强基线。",
    "index.results.model": "模型",
    "index.results.avg": "平均",
    "index.results.mavg": "宏平均",
    "index.results.proprietary": "闭源模型",
    "index.results.open_source": "开源模型",
    "index.results.ours": "WIA",
    "index.resources.kicker": "资源",
    "index.resources.dataset.title": "数据集",
    "index.resources.dataset.desc": "WIA-Corpus 与 WIA-Bench 资源入口。",
    "index.resources.model.title": "模型",
    "index.resources.model.desc": "WIA-WM 模型权重与 checkpoint 入口。",
    "index.resources.code.title": "代码",
    "index.resources.code.desc": "训练、评测与部署代码入口。",
    "index.resources.paper.title": "论文",
    "index.resources.paper.desc": "论文入口",
    "index.resources.paper.link": "技术报告",
    "index.cta.title": "现在就体验 WIA",
    "index.cta.desc": "观看七大数字环境的演示，探索 WIA 的数据、模型与研究成果。",
    "index.footer": "WIA · 世界智能体",
    "nav.demos": "Demo",
    "demos.title": "Demo · WIA",
    "demos.eyebrow": "演示画廊",
    "demos.heading": "七大领域演示",
    "demos.domain_aria": "领域选择",
    "demos.download": "下载视频",
    "demos.domain.android.title": "移动应用界面操作",
    "demos.domain.android.desc": "点击、输入、滑动与 accessibility tree 状态演化。",
    "demos.domain.mcp.title": "MCP 企业工具调用",
    "demos.domain.mcp.desc": "schema 查询、业务记录创建更新删除与 API 返回。",
    "demos.domain.os.title": "桌面操作系统操作",
    "demos.domain.os.desc": "窗口、文件与系统任务交互。",
    "demos.domain.search.title": "网页搜索与浏览",
    "demos.domain.search.desc": "检索、打开链接、阅读页面与结果排序。",
    "demos.domain.swe.title": "软件工程任务",
    "demos.domain.swe.desc": "代码仓库、命令与验证反馈。",
    "demos.domain.terminal.title": "终端环境",
    "demos.domain.terminal.desc": "shell 命令、文件系统与脚本执行。",
    "demos.domain.web.title": "WebArena 网页任务",
    "demos.domain.web.desc": "站点导航、表单与电商/信息任务。",
    "nav.view_demos": "观看 Demo",
    "nav.back_home": "返回首页 →",
    "index.hero.explore": "探索资源"
  },
  "en": {
    "common.switch_language": "Switch language",
    "common.switch_to_zh": "中文",
    "common.switch_to_en": "EN",
    "common.on": "On",
    "common.off": "Off",
    "nav.home": "Home",
    "nav.capabilities": "Capabilities",
    "nav.resources": "Resources",
    "nav.data": "Data",
    "nav.code": "Code",
    "nav.paper": "Paper",
    "nav.method": "Method",
    "nav.results": "Results",
    "index.title": "WIA · World Intelligence Agent",
    "index.meta_description": "WIA is a foundational world model for digital agents, unifying seven digital environments and connecting prediction with action.",
    "index.hero.eyebrow": "Foundation World Model for Digital Agents",
    "index.hero.title": "World<br /><span class=\"grad\">Intelligence Agent</span>",
    "index.hero.subtitle": "Let digital agents understand the environment before taking action. WIA unifies seven digital worlds and uses a foundational world model for state prediction, simulated training, and proactive decisions.",
    "index.stat.cpt_samples": "CPT interaction samples",
    "index.stat.sft_trajectories": "SFT trajectories",
    "index.stat.rl_samples": "RL samples",
    "index.stat.digital_domains": "digital environment domains",
    "index.features.kicker": "Core Building Blocks",
    "index.features.title": "From World Understanding to World Action",
    "index.features.subtitle": "WIA is not a passive simulator; it extends environment prediction into full agent-policy optimization.",
    "index.feature.corpus.title": "WIA-Corpus",
    "index.feature.corpus.desc": "Full-lifecycle CPT, SFT, and RL training corpora that unify heterogeneous digital interactions into action–observation trajectories.",
    "index.feature.bench.title": "WIA-Bench",
    "index.feature.bench.desc": "The first paired ID/OOD world-model benchmark with 1.8k in-domain and 1.7k out-of-domain samples across seven digital environments.",
    "index.feature.wm.title": "WIA-WM",
    "index.feature.wm.desc": "4B / 9B / 27B / 35B-A3B / 122B-A10B foundation world models for systematic state transitions, simulated SFT, simulated RL, and lookahead reasoning.",
    "index.feature.wam.title": "WIA-WAM",
    "index.feature.wam.desc": "Couples prediction and execution through Dual-Perspective Inversion and CSPO so policies implicitly internalize environment dynamics.",
    "index.feature.generalization.title": "Cross-domain Generalization",
    "index.feature.generalization.desc": "Unified modeling across Android, MCP, OS, Search, SWE, Terminal, and Web, without siloing knowledge by interface.",
    "index.feature.scaling.title": "Scaling Roadmap",
    "index.feature.scaling.desc": "From small models to the 122B-A10B MoE large model, validating that world-model capability scales with size for agent training and inference.",
    "index.matrix.kicker": "Model Family",
    "index.matrix.title": "World Model Matrix",
    "index.matrix.subtitle": "A unified model-family lineup covering efficient small models to large-scale MoE.",
    "index.matrix.4b.desc": "Efficient world-model baseline",
    "index.matrix.9b.desc": "Balanced reasoning and capability",
    "index.matrix.27b.desc": "High-performance dense tier",
    "index.matrix.35b.desc": "Efficient MoE scaling",
    "index.matrix.122b.desc": "Large-scale MoE",
    "index.method.kicker": "Pipeline",
    "index.method.title": "Data Infrastructure · World Simulation · Action Execution",
    "index.method.subtitle": "WIA covers the full loop from data infrastructure, world simulation, to action execution: WIA-Corpus and WIA-Bench establish the data/evaluation foundation, WIA-FWM models state transitions and reward dynamics, and WIA-WAM aligns action generation with predicted dynamics.",
    "index.method.unify.title": "Unify Environments",
    "index.method.unify.desc": "Serialize seven digital interfaces into a shared textual action–observation representation.",
    "index.method.train.title": "Train WIA-WM",
    "index.method.train.desc": "CPT → SFT → RL three-stage training builds systematic next-observation prediction.",
    "index.method.sim.title": "Simulate at Scale",
    "index.method.sim.desc": "Use the world model to generate simulated data, provide simulated rewards, and reason ahead.",
    "index.method.act.title": "Couple Actions",
    "index.method.act.desc": "DPI aligns dual-perspective representations, and CSPO directly links action choices with simulated outcomes.",
    "index.domains.kicker": "Seven Domains",
    "index.domains.title": "One model covering the main battlefields of digital agents",
    "index.results.kicker": "Results",
    "index.results.title": "Measuring World Understanding with One Benchmark",
    "index.results.subtitle": "WIA-WM achieves leading average world-model scores on WIA-Bench and significantly outperforms strong baselines on the OOD suite.",
    "index.results.model": "Model",
    "index.results.avg": "Avg",
    "index.results.mavg": "MAvg",
    "index.results.proprietary": "Proprietary",
    "index.results.open_source": "Open-source",
    "index.results.ours": "Ours",
    "index.resources.kicker": "Resources",
    "index.resources.dataset.title": "Dataset",
    "index.resources.dataset.desc": "Entry point for WIA-Corpus and WIA-Bench resources.",
    "index.resources.model.title": "Models",
    "index.resources.model.desc": "Entry point for WIA-WM model weights and checkpoints.",
    "index.resources.code.title": "Code",
    "index.resources.code.desc": "Entry point for training, evaluation, and deployment code.",
    "index.resources.paper.title": "Paper",
    "index.resources.paper.desc": "Paper link",
    "index.resources.paper.link": "Technical Report",
    "index.cta.title": "Experience WIA Now",
    "index.cta.desc": "Watch demonstrations across seven digital environments and explore WIA data, models, and research.",
    "index.footer": "WIA · World Intelligence Agent",
    "nav.demos": "Demo",
    "demos.title": "Demo · WIA",
    "demos.eyebrow": "Demo Gallery",
    "demos.heading": "Seven Domain Demos",
    "demos.domain_aria": "Domain selection",
    "demos.download": "Download Video",
    "demos.domain.android.title": "Mobile App Interaction",
    "demos.domain.android.desc": "Tap, type, scroll, and accessibility-tree state transitions.",
    "demos.domain.mcp.title": "MCP Business Tool Calls",
    "demos.domain.mcp.desc": "Schema queries, create/update/delete records, and API responses.",
    "demos.domain.os.title": "Desktop OS Interaction",
    "demos.domain.os.desc": "Windows, files, and system task interactions.",
    "demos.domain.search.title": "Web Search and Browsing",
    "demos.domain.search.desc": "Retrieval, opening links, page reading, and result ranking.",
    "demos.domain.swe.title": "Software Engineering Tasks",
    "demos.domain.swe.desc": "Code repositories, commands, and verification feedback.",
    "demos.domain.terminal.title": "Terminal Environment",
    "demos.domain.terminal.desc": "Shell commands, filesystem operations, and script execution.",
    "demos.domain.web.title": "WebArena Web Tasks",
    "demos.domain.web.desc": "Site navigation, forms, and e-commerce/information tasks.",
    "nav.view_demos": "Watch Demos",
    "nav.back_home": "Back to Home →",
    "index.hero.explore": "Explore Resources"
  }
};
  function normalizeLang(lang) {
    return lang === "en" ? "en" : "zh";
  }

  function getLang() {
    try {
      return normalizeLang(localStorage.getItem(STORAGE_KEY) || DEFAULT_LANG);
    } catch (e) {
      return DEFAULT_LANG;
    }
  }

  let currentLang = getLang();

  function replaceVars(value, vars) {
    return String(value || "").replace(/\{(\w+)\}/g, (_, key) =>
      vars && Object.prototype.hasOwnProperty.call(vars, key) ? String(vars[key]) : ""
    );
  }

  function t(key, vars) {
    if (!key) return "";
    const table = MESSAGES[currentLang] || MESSAGES[DEFAULT_LANG];
    const value = table[key] || (MESSAGES[DEFAULT_LANG] && MESSAGES[DEFAULT_LANG][key]) || key;
    return replaceVars(value, vars);
  }

  function norm(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  const reverse = {
    zh: new Map(),
    en: new Map()
  };

  Object.keys(MESSAGES.zh).forEach(key => {
    reverse.en.set(norm(MESSAGES.zh[key]), key);
  });
  Object.keys(MESSAGES.en).forEach(key => {
    reverse.zh.set(norm(MESSAGES.en[key]), key);
  });

  const SKIP_SELECTOR = [
    "script",
    "style",
    "textarea",
    "input",
    "select",
    "code",
    "[data-i18n]",
    "[data-i18n-html]",
    "[data-i18n-skip]"
  ].join(",");

  function findReverseKey(lang, text) {
    const map = reverse[lang];
    return map ? map.get(norm(text)) : null;
  }

  function translateStatic(root) {
    const base = root === document ? document : root;
    if (!base) return;

    if (root === document) {
      const html = document.documentElement;
      if (html) {
        html.setAttribute("lang", currentLang === "zh" ? "zh-CN" : "en");
        if (html.dataset.i18nTitleKey) {
          document.title = t(html.dataset.i18nTitleKey);
        }
      }
      document.querySelectorAll("meta[data-i18n-content]").forEach(meta => {
        meta.setAttribute("content", t(meta.dataset.i18nContent));
      });
    }

    base.querySelectorAll("[data-i18n]").forEach(el => {
      el.textContent = t(el.dataset.i18n);
    });

    base.querySelectorAll("[data-i18n-html]").forEach(el => {
      el.innerHTML = t(el.dataset.i18nHtml);
    });

    base.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
      el.setAttribute("placeholder", t(el.dataset.i18nPlaceholder));
    });

    base.querySelectorAll("[data-i18n-aria-label]").forEach(el => {
      el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel));
    });

    base.querySelectorAll("input[placeholder], textarea[placeholder]").forEach(el => {
      const key = findReverseKey(currentLang, el.getAttribute("placeholder"));
      if (key) el.setAttribute("placeholder", t(key));
    });

    const walker = document.createTreeWalker(
      root === document ? document.body : root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node || !node.nodeValue) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent || parent.closest(SKIP_SELECTOR)) return NodeFilter.FILTER_REJECT;
          const text = norm(node.nodeValue);
          if (!text) return NodeFilter.FILTER_REJECT;
          return findReverseKey(currentLang, text) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      }
    );

    const replacements = [];
    let node;
    while ((node = walker.nextNode())) {
      const key = findReverseKey(currentLang, node.nodeValue);
      if (key) replacements.push([node, key]);
    }
    replacements.forEach(([node, key]) => {
      node.nodeValue = t(key);
    });
  }

  function ensureLanguageSwitch() {
    document.querySelectorAll(".nav-actions").forEach(actions => {
      if (actions.querySelector(".lang-switch")) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "lang-switch btn btn-outline-light btn-sm";
      button.dataset.i18nAriaLabel = "common.switch_language";
      button.title = t("common.switch_language");
      button.setAttribute("aria-label", t("common.switch_language"));
      actions.appendChild(button);
    });
    syncLanguageSwitch();
  }

  function syncLanguageSwitch() {
    document.querySelectorAll(".lang-switch").forEach(button => {
      const label = currentLang === "zh" ? t("common.switch_to_en") : t("common.switch_to_zh");
      button.textContent = label;
      button.title = t("common.switch_language");
      button.setAttribute("aria-label", t("common.switch_language"));
    });
  }

  function setLang(lang) {
    const next = normalizeLang(lang);
    if (next === currentLang) return currentLang;
    currentLang = next;
    try {
      localStorage.setItem(STORAGE_KEY, currentLang);
    } catch (e) {}
    translateStatic(document);
    syncLanguageSwitch();
    window.dispatchEvent(new CustomEvent("i18n:change", { detail: { lang: currentLang } }));
    return currentLang;
  }

  document.addEventListener("click", event => {
    const button = event.target.closest(".lang-switch");
    if (!button) return;
    setLang(currentLang === "zh" ? "en" : "zh");
  });

  window.I18N = {
    getLang: getLang,
    setLang: setLang,
    t: t,
    applyStatic: translateStatic,
    ensureLanguageSwitch: ensureLanguageSwitch
  };

  if (document.body) {
    translateStatic(document);
    ensureLanguageSwitch();
  } else {
    document.addEventListener("DOMContentLoaded", () => {
      translateStatic(document);
      ensureLanguageSwitch();
    });
  }
})();
