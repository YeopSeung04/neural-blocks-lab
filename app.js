import { TrainingSession } from "./ml-core.mjs";
import {
  LAYER_CATALOG,
  MODEL_FAMILIES,
  cloneTemplate,
  createLayer,
  formatShape,
  validateArchitecture,
} from "./layer-catalog.mjs";
import {
  buildClassifier,
  buildGan,
} from "./tf-models.mjs";
import {
  GanTrainingSession,
  TfClassifierSession,
} from "./tf-training.mjs";

const elements = {
  modelFamily: document.getElementById("modelFamily"),
  engineBadge: document.getElementById("engineBadge"),
  familyDescription: document.getElementById("familyDescription"),
  branchTabs: document.getElementById("branchTabs"),
  layerPalette: document.getElementById("layerPalette"),
  networkStack: document.getElementById("networkStack"),
  hiddenLayerTemplate: document.getElementById("hiddenLayerTemplate"),
  toggleTraining: document.getElementById("toggleTraining"),
  stepTraining: document.getElementById("stepTraining"),
  resetTraining: document.getElementById("resetTraining"),
  trainingStatus: document.getElementById("trainingStatus"),
  statusDot: document.getElementById("statusDot"),
  datasetType: document.getElementById("datasetType"),
  dataCount: document.getElementById("dataCount"),
  dataCountValue: document.getElementById("dataCountValue"),
  noise: document.getElementById("noise"),
  noiseValue: document.getElementById("noiseValue"),
  optimizer: document.getElementById("optimizer"),
  learningRate: document.getElementById("learningRate"),
  learningRateValue: document.getElementById("learningRateValue"),
  batchSize: document.getElementById("batchSize"),
  batchSizeValue: document.getElementById("batchSizeValue"),
  trainingSpeed: document.getElementById("trainingSpeed"),
  speedValue: document.getElementById("speedValue"),
  maxEpochs: document.getElementById("maxEpochs"),
  maxEpochsValue: document.getElementById("maxEpochsValue"),
  parameterCount: document.getElementById("parameterCount"),
  compileStatus: document.getElementById("compileStatus"),
  architectureExplanation: document.getElementById("architectureExplanation"),
  decisionCanvas: document.getElementById("decisionCanvas"),
  networkCanvas: document.getElementById("networkCanvas"),
  lossCanvas: document.getElementById("lossCanvas"),
  datasetBadge: document.getElementById("datasetBadge"),
  probeValue: document.getElementById("probeValue"),
  epochBadge: document.getElementById("epochBadge"),
  trainLossLabel: document.getElementById("trainLossLabel"),
  validationLossLabel: document.getElementById("validationLossLabel"),
  trainAccuracyLabel: document.getElementById("trainAccuracyLabel"),
  validationAccuracyLabel: document.getElementById("validationAccuracyLabel"),
  trainLoss: document.getElementById("trainLoss"),
  validationLoss: document.getElementById("validationLoss"),
  trainAccuracy: document.getElementById("trainAccuracy"),
  validationAccuracy: document.getElementById("validationAccuracy"),
  primaryVisualTitle: document.getElementById("primaryVisualTitle"),
  primaryVisualDescription: document.getElementById("primaryVisualDescription"),
  networkVisualTitle: document.getElementById("networkVisualTitle"),
  networkVisualDescription: document.getElementById("networkVisualDescription"),
  lossVisualTitle: document.getElementById("lossVisualTitle"),
  lossVisualDescription: document.getElementById("lossVisualDescription"),
};

const datasetLabels = {
  xor: "XOR",
  circles: "원형 분류",
  linear: "선형 분류",
};
const familyLearningRates = {
  mlp: 0.03,
  cnn: 0.01,
  rnn: 0.01,
  gan: 0.002,
};

let hiddenLayers = [
  { id: crypto.randomUUID(), units: 8, activation: "tanh" },
];
let session;
let running = false;
let probe = [0.2, -0.2];
let lastVisualUpdate = 0;
let lastAdvancedTrainingAt = 0;
let draggedLayerId = null;
let activeGanBranch = "generator";
let compiledAdvancedModels = null;
let advancedCompileInfo = null;
let advancedTrainingSession = null;
let advancedTrainingPending = false;
const advancedArchitectures = {
  cnn: cloneTemplate("cnn"),
  rnn: cloneTemplate("rnn"),
  gan: cloneTemplate("gan"),
};

function isAdvancedFamily() {
  return elements.modelFamily.value !== "mlp";
}

function activeAdvancedLayers() {
  const family = elements.modelFamily.value;
  if (family === "gan") return advancedArchitectures.gan[activeGanBranch];
  return advancedArchitectures[family];
}

function currentConfig() {
  return {
    dataset: elements.datasetType.value,
    count: Number(elements.dataCount.value),
    noise: Number(elements.noise.value),
    hiddenLayers: hiddenLayers.map(({ units, activation }) => ({ units, activation })),
    optimizer: elements.optimizer.value,
    learningRate: Number(elements.learningRate.value),
    batchSize: Number(elements.batchSize.value),
    seed: 23,
  };
}

function createFixedBlock(type, title, detail) {
  const block = document.createElement("article");
  block.className = `network-block fixed ${type}`;
  block.innerHTML = `<div><strong>${title}</strong><div class="muted">${detail}</div></div>`;
  return block;
}

function advancedInputShape() {
  const family = elements.modelFamily.value;
  if (family === "gan") return activeGanBranch === "generator" ? [4] : [2];
  return MODEL_FAMILIES[family].inputShape;
}

function advancedOutputLabel() {
  if (elements.modelFamily.value !== "gan") return "Dense 1 + Sigmoid";
  return activeGanBranch === "generator"
    ? "Dense 2 + Tanh (generated point)"
    : "Dense 1 + Sigmoid (real/fake)";
}

function createFieldControl(layer, fieldName, field) {
  const label = document.createElement("label");
  label.textContent = field.label;
  let control;

  if (field.type === "select") {
    control = document.createElement("select");
    for (const optionValue of field.options) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue;
      control.append(option);
    }
    control.value = String(layer[fieldName]);
  } else if (field.type === "boolean") {
    control = document.createElement("select");
    control.innerHTML = '<option value="false">False</option><option value="true">True</option>';
    control.value = String(Boolean(layer[fieldName]));
  } else {
    control = document.createElement("input");
    control.type = field.type === "text" ? "text" : "number";
    if (field.min !== undefined) control.min = String(field.min);
    if (field.max !== undefined) control.max = String(field.max);
    if (field.step !== undefined) control.step = String(field.step);
    control.value = String(layer[fieldName]);
  }

  control.dataset.field = fieldName;
  label.append(control);
  return label;
}

function renderAdvancedArchitecture() {
  const layers = activeAdvancedLayers();
  const validation = validateArchitecture(advancedInputShape(), layers);
  elements.networkStack.replaceChildren();
  elements.networkStack.append(
    createFixedBlock("input", "Input", `shape ${formatShape(advancedInputShape())}`),
  );

  layers.forEach((layer, index) => {
    const definition = LAYER_CATALOG[layer.type];
    const block = document.createElement("article");
    block.className = `network-block advanced-block${validation.rows[index]?.valid === false ? " invalid" : ""}`;
    block.dataset.layerId = layer.id;
    block.draggable = true;

    const handle = document.createElement("div");
    handle.className = "drag-handle";
    handle.textContent = "::";

    const main = document.createElement("div");
    main.className = "block-main";
    const title = document.createElement("strong");
    title.textContent = definition.label;
    const fields = document.createElement("div");
    fields.className = "advanced-fields";
    for (const [fieldName, field] of Object.entries(definition.fields)) {
      fields.append(createFieldControl(layer, fieldName, field));
    }
    const shape = document.createElement("div");
    shape.className = "shape-row";
    const shapeRow = validation.rows[index];
    shape.textContent = shapeRow?.valid
      ? `${formatShape(shapeRow.input)} -> ${formatShape(shapeRow.output)}`
      : `${formatShape(shapeRow?.input)} -> ${shapeRow?.error ?? "invalid"}`;
    main.append(title, fields, shape);

    const actions = document.createElement("div");
    actions.className = "block-actions";
    actions.innerHTML =
      `<button class="move-up" type="button"${index === 0 ? " disabled" : ""}>Up</button>` +
      `<button class="move-down" type="button"${index === layers.length - 1 ? " disabled" : ""}>Down</button>` +
      '<button class="remove-layer danger" type="button">삭제</button>';

    block.append(handle, main, actions);
    elements.networkStack.append(block);
  });

  if (!layers.length) {
    const hint = document.createElement("div");
    hint.className = "drop-hint";
    hint.textContent = "팔레트에서 레이어를 추가하세요.";
    elements.networkStack.append(hint);
  }

  elements.networkStack.append(
    createFixedBlock("output", "Output head", advancedOutputLabel()),
  );
}

function renderArchitecture() {
  if (isAdvancedFamily()) {
    renderAdvancedArchitecture();
    updateArchitectureDescription();
    return;
  }

  elements.networkStack.replaceChildren();
  elements.networkStack.append(
    createFixedBlock("input", "Input", "2 features: x1, x2"),
  );

  hiddenLayers.forEach((layer, index) => {
    const block = elements.hiddenLayerTemplate.content.firstElementChild.cloneNode(true);
    block.dataset.layerId = layer.id;
    block.dataset.activation = layer.activation;
    block.querySelector(".units-input").value = String(layer.units);
    block.querySelector(".activation-select").value = layer.activation;
    block.querySelector(".move-up").disabled = index === 0;
    block.querySelector(".move-down").disabled = index === hiddenLayers.length - 1;
    elements.networkStack.append(block);
  });

  if (hiddenLayers.length === 0) {
    const hint = document.createElement("div");
    hint.className = "drop-hint";
    hint.textContent = "여기에 블록을 추가하면 비선형 신경망이 됩니다.";
    elements.networkStack.append(hint);
  }

  elements.networkStack.append(
    createFixedBlock("output", "Output", "1 neuron + Sigmoid"),
  );

  updateArchitectureDescription();
}

function parameterCount() {
  let total = 0;
  let previousUnits = 2;
  for (const layer of hiddenLayers) {
    total += previousUnits * layer.units + layer.units;
    previousUnits = layer.units;
  }
  total += previousUnits + 1;
  return total;
}

function renderPalette() {
  elements.layerPalette.replaceChildren();
  if (!isAdvancedFamily()) {
    const presets = [
      ["relu", "Dense + ReLU", "복잡한 패턴에 강한 기본 블록"],
      ["tanh", "Dense + Tanh", "-1부터 1 사이의 부드러운 활성화"],
      ["sigmoid", "Dense + Sigmoid", "확률 형태로 값을 압축하는 블록"],
    ];
    for (const [activation, title, description] of presets) {
      const button = document.createElement("button");
      button.className = `palette-block ${activation}`;
      button.type = "button";
      button.draggable = true;
      button.dataset.activation = activation;
      button.innerHTML = `<strong>${title}</strong><small>${description}</small>`;
      elements.layerPalette.append(button);
    }
    return;
  }

  const family = MODEL_FAMILIES[elements.modelFamily.value];
  for (const layerType of family.palette) {
    const definition = LAYER_CATALOG[layerType];
    const button = document.createElement("button");
    button.className = "palette-block catalog";
    button.type = "button";
    button.draggable = true;
    button.dataset.layerType = layerType;
    button.dataset.category = definition.category;
    button.innerHTML = `<strong>${definition.label}</strong><small>${definition.description}</small>`;
    elements.layerPalette.append(button);
  }
}

function disposeCompiledAdvancedModels() {
  if (!compiledAdvancedModels) return;
  const ownedByTrainingSession = advancedTrainingSession && (
    compiledAdvancedModels === advancedTrainingSession.model ||
    (
      Boolean(compiledAdvancedModels.generator) &&
      compiledAdvancedModels.generator === advancedTrainingSession.generator &&
      compiledAdvancedModels.discriminator === advancedTrainingSession.discriminator
    )
  );
  if (ownedByTrainingSession) {
    compiledAdvancedModels = null;
    return;
  }
  if (compiledAdvancedModels.generator) {
    compiledAdvancedModels.generator.dispose();
    compiledAdvancedModels.discriminator.dispose();
  } else {
    compiledAdvancedModels.dispose();
  }
  compiledAdvancedModels = null;
}

function disposeAdvancedTrainingSession() {
  advancedTrainingSession?.dispose?.();
  advancedTrainingSession = null;
  advancedTrainingPending = false;
}

function createAdvancedTrainingSession() {
  disposeAdvancedTrainingSession();
  if (advancedCompileInfo?.error) return;
  const tf = globalThis.tf;
  const family = elements.modelFamily.value;
  const config = {
    count: Number(elements.dataCount.value),
    optimizer: elements.optimizer.value,
    learningRate: Number(elements.learningRate.value),
    batchSize: Number(elements.batchSize.value),
  };
  advancedTrainingSession = family === "gan"
    ? new GanTrainingSession(tf, advancedArchitectures.gan, {
        ...config,
        models: compiledAdvancedModels,
      })
    : new TfClassifierSession(tf, family, activeAdvancedLayers(), {
        ...config,
        model: compiledAdvancedModels,
      });
}

function compileAdvancedArchitecture() {
  const tf = globalThis.tf;
  const family = elements.modelFamily.value;
  const layers = activeAdvancedLayers();
  const validation = validateArchitecture(advancedInputShape(), layers);
  disposeCompiledAdvancedModels();
  advancedCompileInfo = {
    family,
    validation,
    parameterCount: 0,
    output: null,
    error: null,
  };

  try {
    if (!tf) throw new Error("TensorFlow.js failed to load");
    if (family === "gan") {
      compiledAdvancedModels = buildGan(tf, advancedArchitectures.gan);
      advancedCompileInfo.parameterCount =
        compiledAdvancedModels.generator.countParams() +
        compiledAdvancedModels.discriminator.countParams();
      advancedCompileInfo.output =
        `G [batch, 4] -> [batch, 2], D [batch, 2] -> [batch, 1]`;
    } else {
      compiledAdvancedModels = buildClassifier(tf, family, layers);
      advancedCompileInfo.parameterCount = compiledAdvancedModels.countParams();
      advancedCompileInfo.output =
        `[batch, ${MODEL_FAMILIES[family].inputShape.join(", ")}] -> [batch, 1]`;
    }
  } catch (error) {
    advancedCompileInfo.error = error.message;
  }

  elements.parameterCount.textContent =
    `${advancedCompileInfo.parameterCount.toLocaleString()} parameters`;
  elements.compileStatus.className =
    `compile-status ${advancedCompileInfo.error ? "invalid" : "valid"}`;
  elements.compileStatus.textContent = advancedCompileInfo.error
    ? `Compile error: ${advancedCompileInfo.error}`
    : `TensorFlow.js compile OK | ${advancedCompileInfo.output}`;
  try {
    createAdvancedTrainingSession();
  } catch (error) {
    advancedCompileInfo.error = error.message;
    disposeAdvancedTrainingSession();
    elements.compileStatus.className = "compile-status invalid";
    elements.compileStatus.textContent = `Training session error: ${error.message}`;
  }
}

function updateArchitectureDescription() {
  if (isAdvancedFamily()) {
    const family = elements.modelFamily.value;
    const modelName = family === "gan"
      ? `GAN ${activeGanBranch} branch`
      : MODEL_FAMILIES[family].label;
    const objectParticle = family === "cnn" ? "을" : "를";
    elements.architectureExplanation.textContent =
      `${modelName}${objectParticle} TensorFlow.js Layers 모델로 컴파일합니다. ` +
      "각 블록의 input/output shape가 연결되지 않으면 해당 레이어가 빨간색으로 표시됩니다.";
    return;
  }

  const total = parameterCount();
  elements.parameterCount.textContent = `${total.toLocaleString()} parameters`;

  if (hiddenLayers.length === 0) {
    elements.architectureExplanation.textContent =
      "현재 모델은 은닉층이 없는 로지스틱 회귀입니다. 선형 데이터는 학습할 수 있지만 XOR이나 원형 데이터의 비선형 경계는 표현할 수 없습니다.";
    return;
  }

  const layerText = hiddenLayers
    .map((layer) => `${layer.units} ${layer.activation.toUpperCase()}`)
    .join(" -> ");
  elements.architectureExplanation.textContent =
    `입력 2개가 ${layerText} 은닉층을 통과한 뒤 Sigmoid 출력으로 분류됩니다. ` +
    `블록을 추가하거나 삭제하면 가중치 ${total.toLocaleString()}개가 새로 초기화되고 학습이 처음부터 시작됩니다.`;
}

function rebuildAdvancedStudio() {
  running = false;
  updateStatus();
  updateControlLabels();
  renderPalette();
  renderArchitecture();
  compileAdvancedArchitecture();
  setAdvancedEngineState();
  drawAdvancedVisuals();
}

function addAdvancedLayer(type, targetIndex = activeAdvancedLayers().length) {
  const layers = activeAdvancedLayers();
  layers.splice(Math.max(0, Math.min(targetIndex, layers.length)), 0, createLayer(type));
  rebuildAdvancedStudio();
}

function removeAdvancedLayer(layerId) {
  const layers = activeAdvancedLayers();
  const index = layers.findIndex((layer) => layer.id === layerId);
  if (index >= 0) layers.splice(index, 1);
  rebuildAdvancedStudio();
}

function moveAdvancedLayer(layerId, direction) {
  const layers = activeAdvancedLayers();
  const index = layers.findIndex((layer) => layer.id === layerId);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= layers.length) return;
  [layers[index], layers[target]] = [layers[target], layers[index]];
  rebuildAdvancedStudio();
}

function moveAdvancedLayerTo(layerId, targetId) {
  const layers = activeAdvancedLayers();
  const sourceIndex = layers.findIndex((layer) => layer.id === layerId);
  const targetIndex = layers.findIndex((layer) => layer.id === targetId);
  if (sourceIndex < 0 || targetIndex < 0 || sourceIndex === targetIndex) return;
  const [layer] = layers.splice(sourceIndex, 1);
  layers.splice(targetIndex, 0, layer);
  rebuildAdvancedStudio();
}

function addLayer(activation = "relu", targetIndex = hiddenLayers.length) {
  if (hiddenLayers.length >= 6) {
    setStatus("은닉층은 MVP에서 최대 6개입니다.", false);
    return;
  }
  const layer = {
    id: crypto.randomUUID(),
    units: 8,
    activation,
  };
  hiddenLayers.splice(Math.max(0, Math.min(targetIndex, hiddenLayers.length)), 0, layer);
  rebuildModel();
}

function removeLayer(layerId) {
  hiddenLayers = hiddenLayers.filter((layer) => layer.id !== layerId);
  rebuildModel();
}

function moveLayer(layerId, direction) {
  const index = hiddenLayers.findIndex((layer) => layer.id === layerId);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= hiddenLayers.length) return;
  [hiddenLayers[index], hiddenLayers[target]] = [hiddenLayers[target], hiddenLayers[index]];
  rebuildModel();
}

function moveLayerTo(layerId, targetId) {
  if (!layerId || !targetId || layerId === targetId) return;
  const sourceIndex = hiddenLayers.findIndex((layer) => layer.id === layerId);
  const targetIndex = hiddenLayers.findIndex((layer) => layer.id === targetId);
  if (sourceIndex < 0 || targetIndex < 0) return;
  const [layer] = hiddenLayers.splice(sourceIndex, 1);
  hiddenLayers.splice(targetIndex, 0, layer);
  rebuildModel();
}

function updateControlLabels() {
  elements.dataCountValue.textContent = elements.dataCount.value;
  elements.noiseValue.textContent = Number(elements.noise.value).toFixed(2);
  elements.learningRateValue.textContent = Number(elements.learningRate.value).toFixed(3);
  elements.batchSizeValue.textContent = elements.batchSize.value;
  elements.speedValue.textContent = elements.trainingSpeed.value;
  elements.maxEpochsValue.textContent = elements.maxEpochs.value;
  elements.datasetBadge.textContent = datasetLabels[elements.datasetType.value];
  const family = MODEL_FAMILIES[elements.modelFamily.value];
  elements.familyDescription.textContent = family.description;
}

function rebuildModel({ preserveRunning = false } = {}) {
  if (isAdvancedFamily()) {
    rebuildAdvancedStudio();
    return;
  }

  disposeCompiledAdvancedModels();
  disposeAdvancedTrainingSession();
  advancedCompileInfo = null;
  const wasRunning = preserveRunning && running;
  session = new TrainingSession(currentConfig());
  running = wasRunning;
  elements.primaryVisualTitle.textContent = "실시간 결정경계";
  elements.primaryVisualDescription.textContent = "배경색은 모델 예측, 점은 학습 데이터입니다.";
  elements.networkVisualTitle.textContent = "뉴런 활성화";
  elements.networkVisualDescription.textContent = "결정경계를 클릭해 입력점을 바꿀 수 있습니다.";
  elements.lossVisualTitle.textContent = "학습 곡선";
  elements.lossVisualDescription.textContent = "Train / validation loss";
  elements.trainLossLabel.textContent = "Train loss";
  elements.validationLossLabel.textContent = "Val loss";
  elements.trainAccuracyLabel.textContent = "Train acc";
  elements.validationAccuracyLabel.textContent = "Val acc";
  elements.probeValue.textContent = `x: ${probe[0].toFixed(2)}, y: ${probe[1].toFixed(2)}`;
  renderArchitecture();
  updateControlLabels();
  setAdvancedEngineState();
  elements.compileStatus.className = "compile-status valid";
  elements.compileStatus.textContent = "Custom JavaScript backpropagation engine ready";
  updateStatus();
  drawAll();
}

function setAdvancedEngineState() {
  const advanced = isAdvancedFamily();
  const family = elements.modelFamily.value;
  const datasetOptions = [...elements.datasetType.options];
  elements.engineBadge.textContent = advanced ? "TF.js live engine" : "Live engine";
  elements.branchTabs.hidden = family !== "gan";
  elements.branchTabs.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.branch === activeGanBranch);
  });
  elements.toggleTraining.disabled = advanced && Boolean(advancedCompileInfo?.error);
  elements.stepTraining.disabled = advanced && Boolean(advancedCompileInfo?.error);
  elements.datasetType.disabled = advanced;
  elements.noise.disabled = advanced;
  elements.dataCount.disabled = advanced && family === "gan";
  datasetOptions.forEach((option, index) => {
    option.hidden = advanced && index > 0;
  });
  datasetOptions[0].textContent = advanced
    ? {
        cnn: "8x8 선 방향 이미지",
        rnn: "12-step 상승/하락 시계열",
        gan: "2D 원형 목표 분포",
      }[family]
    : "XOR";
  if (advanced) elements.datasetType.selectedIndex = 0;
  for (const control of [
    elements.optimizer,
    elements.learningRate,
    elements.batchSize,
    elements.trainingSpeed,
    elements.maxEpochs,
  ]) control.disabled = false;
  if (advanced) {
    setStatus(advancedCompileInfo?.error ? "컴파일 오류" : "학습 준비", false);
  }
}

function setStatus(text, active = running) {
  elements.trainingStatus.textContent = text;
  elements.statusDot.classList.toggle("running", active);
}

function updateStatus() {
  elements.toggleTraining.textContent = running ? "일시정지" : "학습 시작";
  setStatus(running ? "학습 중" : "일시정지", running);
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const pixelWidth = Math.max(1, Math.floor(rect.width * ratio));
  const pixelHeight = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function probabilityColor(probability) {
  const p = Math.max(0, Math.min(1, probability));
  const low = [37, 79, 122];
  const high = [151, 68, 48];
  return `rgb(${Math.round(low[0] * (1 - p) + high[0] * p)}, ` +
    `${Math.round(low[1] * (1 - p) + high[1] * p)}, ` +
    `${Math.round(low[2] * (1 - p) + high[2] * p)})`;
}

function drawDecisionBoundary() {
  const { context, width, height } = resizeCanvas(elements.decisionCanvas);
  context.clearRect(0, 0, width, height);

  const grid = Math.max(34, Math.min(64, Math.floor(width / 10)));
  const cellWidth = width / grid;
  const cellHeight = height / grid;
  for (let yIndex = 0; yIndex < grid; yIndex += 1) {
    const y = 1.2 - (yIndex + 0.5) / grid * 2.4;
    for (let xIndex = 0; xIndex < grid; xIndex += 1) {
      const x = -1.2 + (xIndex + 0.5) / grid * 2.4;
      const probability = session.network.predict([x, y]);
      context.fillStyle = probabilityColor(probability);
      context.fillRect(
        xIndex * cellWidth,
        yIndex * cellHeight,
        Math.ceil(cellWidth) + 1,
        Math.ceil(cellHeight) + 1,
      );
    }
  }

  const toCanvas = ([x, y]) => ({
    x: (x + 1.2) / 2.4 * width,
    y: (1.2 - y) / 2.4 * height,
  });

  context.strokeStyle = "rgba(255,255,255,0.2)";
  context.lineWidth = 1;
  const origin = toCanvas([0, 0]);
  context.beginPath();
  context.moveTo(origin.x, 0);
  context.lineTo(origin.x, height);
  context.moveTo(0, origin.y);
  context.lineTo(width, origin.y);
  context.stroke();

  const validationSet = new Set(session.validationPoints);
  for (const point of session.points) {
    const position = toCanvas(point.input);
    context.beginPath();
    context.arc(position.x, position.y, validationSet.has(point) ? 4.5 : 3.5, 0, Math.PI * 2);
    context.fillStyle = point.target === 1 ? "#ffbf75" : "#82c5ff";
    context.fill();
    if (validationSet.has(point)) {
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.2;
      context.stroke();
    }
  }

  const probePosition = toCanvas(probe);
  context.strokeStyle = "#57df8b";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(probePosition.x, probePosition.y, 8, 0, Math.PI * 2);
  context.moveTo(probePosition.x - 12, probePosition.y);
  context.lineTo(probePosition.x + 12, probePosition.y);
  context.moveTo(probePosition.x, probePosition.y - 12);
  context.lineTo(probePosition.x, probePosition.y + 12);
  context.stroke();
}

function activationColor(value, activation) {
  if (activation === "sigmoid") {
    const strength = Math.round(50 + Math.max(0, Math.min(1, value)) * 170);
    return `rgb(35, ${strength}, 135)`;
  }
  if (value >= 0) {
    const strength = Math.round(75 + Math.min(1, Math.abs(value)) * 150);
    return `rgb(35, ${strength}, 125)`;
  }
  const strength = Math.round(75 + Math.min(1, Math.abs(value)) * 150);
  return `rgb(${strength}, 80, 55)`;
}

function displayedIndices(total, maximum = 12) {
  if (total <= maximum) return Array.from({ length: total }, (_, index) => index);
  return Array.from({ length: maximum }, (_, index) =>
    Math.round(index * (total - 1) / (maximum - 1)),
  );
}

function drawNetwork() {
  const { context, width, height } = resizeCanvas(elements.networkCanvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f151c";
  context.fillRect(0, 0, width, height);

  const architecture = session.network.architecture();
  const snapshot = session.network.activationSnapshot(probe);
  const columns = architecture.map((layer, layerIndex) => {
    const values = layerIndex === 0 ? probe : snapshot.layers[layerIndex - 1];
    const indices = displayedIndices(layer.units);
    return {
      ...layer,
      values,
      indices,
      x: 46 + layerIndex * (width - 92) / Math.max(1, architecture.length - 1),
    };
  });

  function nodeY(index, count) {
    if (count === 1) return height / 2;
    return 38 + index * (height - 76) / (count - 1);
  }

  for (let layerIndex = 1; layerIndex < columns.length; layerIndex += 1) {
    const previous = columns[layerIndex - 1];
    const current = columns[layerIndex];
    const weightLayer = session.network.layers[layerIndex - 1];
    for (let currentDisplay = 0; currentDisplay < current.indices.length; currentDisplay += 1) {
      const currentIndex = current.indices[currentDisplay];
      for (let previousDisplay = 0; previousDisplay < previous.indices.length; previousDisplay += 1) {
        const previousIndex = previous.indices[previousDisplay];
        const weight = weightLayer.weights[currentIndex][previousIndex];
        const magnitude = Math.min(1, Math.abs(weight) / 2.5);
        context.strokeStyle = weight >= 0
          ? `rgba(57, 208, 199, ${0.12 + magnitude * 0.68})`
          : `rgba(255, 107, 107, ${0.12 + magnitude * 0.68})`;
        context.lineWidth = 0.5 + magnitude * 2.2;
        context.beginPath();
        context.moveTo(previous.x, nodeY(previousDisplay, previous.indices.length));
        context.lineTo(current.x, nodeY(currentDisplay, current.indices.length));
        context.stroke();
      }
    }
  }

  columns.forEach((column) => {
    column.indices.forEach((unitIndex, displayIndex) => {
      const y = nodeY(displayIndex, column.indices.length);
      const value = column.values[unitIndex] ?? 0;
      context.beginPath();
      context.arc(column.x, y, 8, 0, Math.PI * 2);
      context.fillStyle = activationColor(value, column.activation);
      context.fill();
      context.strokeStyle = "rgba(255,255,255,0.65)";
      context.lineWidth = 1;
      context.stroke();
    });

    context.fillStyle = "#c8d3e6";
    context.font = "11px system-ui";
    context.textAlign = "center";
    const label = column.type === "input"
      ? "Input"
      : column.type === "output"
        ? "Output"
        : `${column.units} ${column.activation}`;
    context.fillText(label, column.x, 18);
    if (column.units > column.indices.length) {
      context.fillStyle = "#8291a1";
      context.fillText(`+${column.units - column.indices.length} hidden`, column.x, height - 8);
    }
  });

  context.textAlign = "left";
  context.fillStyle = "#a7b4c2";
  context.font = "11px system-ui";
  context.fillText(`prediction ${(snapshot.output * 100).toFixed(1)}%`, 10, height - 10);
}

function drawLossChart(history = session?.history ?? []) {
  const { context, width, height } = resizeCanvas(elements.lossCanvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f151c";
  context.fillRect(0, 0, width, height);

  if (!history.length) {
    context.fillStyle = "#a7b4c2";
    context.font = "13px system-ui";
    context.fillText("학습을 시작하면 loss가 표시됩니다.", 20, 36);
    return;
  }
  const values = history.flatMap((row) => [row.trainLoss, row.validationLoss]);
  const maxValue = Math.max(0.7, ...values);
  const minValue = Math.min(...values, 0);
  const left = 40;
  const top = 22;
  const right = width - 14;
  const bottom = height - 34;

  context.strokeStyle = "rgba(255,255,255,0.13)";
  context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = top + (bottom - top) * index / 4;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(right, y);
    context.stroke();
  }

  function point(index, value) {
    return {
      x: left + (right - left) * index / Math.max(1, history.length - 1),
      y: bottom - (bottom - top) * (value - minValue) / Math.max(1e-9, maxValue - minValue),
    };
  }

  const ganMode = isAdvancedFamily() && elements.modelFamily.value === "gan";
  const firstSeriesLabel = ganMode ? "Generator" : "Train";
  const secondSeriesLabel = ganMode ? "Discriminator" : "Validation";
  const series = [
    { key: "trainLoss", color: "#ff6b6b" },
    { key: "validationLoss", color: "#57df8b" },
  ];

  for (const item of series) {
    context.strokeStyle = item.color;
    context.lineWidth = 2.5;
    context.beginPath();
    history.forEach((row, index) => {
      const position = point(index, row[item.key]);
      if (index === 0) context.moveTo(position.x, position.y);
      else context.lineTo(position.x, position.y);
    });
    context.stroke();
  }

  context.font = "11px system-ui";
  context.fillStyle = "#ff6b6b";
  context.fillText(firstSeriesLabel, left, height - 10);
  context.fillStyle = "#57df8b";
  context.fillText(secondSeriesLabel, left + (ganMode ? 72 : 54), height - 10);
  context.fillStyle = "#a7b4c2";
  context.textAlign = "right";
  context.fillText(maxValue.toFixed(2), left - 6, top + 3);
  context.fillText(minValue.toFixed(2), left - 6, bottom);
  context.textAlign = "left";
}

function updateMetrics() {
  const metrics = session.metrics();
  elements.trainLoss.textContent = metrics.train.loss.toFixed(4);
  elements.validationLoss.textContent = metrics.validation.loss.toFixed(4);
  elements.trainAccuracy.textContent = `${(metrics.train.accuracy * 100).toFixed(1)}%`;
  elements.validationAccuracy.textContent = `${(metrics.validation.accuracy * 100).toFixed(1)}%`;
  elements.epochBadge.textContent = `Epoch ${metrics.epoch}`;

  const finite = [
    metrics.train.loss,
    metrics.validation.loss,
    metrics.train.accuracy,
    metrics.validation.accuracy,
  ].every(Number.isFinite);
  if (!finite) {
    running = false;
    updateStatus();
    setStatus("학습 발산: learning rate를 낮추세요.", false);
  }
}

function drawTextPanel(canvas, title, lines, accent = "#39d0c7") {
  const { context, width, height } = resizeCanvas(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f151c";
  context.fillRect(0, 0, width, height);
  context.fillStyle = accent;
  context.font = "bold 16px system-ui";
  context.fillText(title, 20, 32);
  context.font = "12px ui-monospace, monospace";
  lines.forEach((line, index) => {
    context.fillStyle = index === 0 ? "#f1f5f9" : "#a7b4c2";
    context.fillText(line, 20, 62 + index * 22);
  });
}

function drawShapePipeline() {
  const { context, width, height } = resizeCanvas(elements.decisionCanvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f151c";
  context.fillRect(0, 0, width, height);

  const layers = activeAdvancedLayers();
  const validation = validateArchitecture(advancedInputShape(), layers);
  const cards = [
    { label: "Input", shape: advancedInputShape(), valid: true },
    ...layers.map((layer, index) => ({
      label: LAYER_CATALOG[layer.type].label,
      shape: validation.rows[index]?.output,
      valid: validation.rows[index]?.valid !== false,
      error: validation.rows[index]?.error,
    })),
    {
      label: "Output head",
      shape: elements.modelFamily.value === "gan" && activeGanBranch === "generator" ? [2] : [1],
      valid: validation.valid,
    },
  ];

  const columns = Math.max(1, Math.min(4, Math.floor(width / 190)));
  const cardWidth = (width - 28 - (columns - 1) * 12) / columns;
  const cardHeight = 92;

  cards.forEach((card, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = 14 + column * (cardWidth + 12);
    const y = 18 + row * (cardHeight + 18);
    context.fillStyle = card.valid ? "#1b2430" : "#3b2022";
    context.strokeStyle = card.valid ? "#42566a" : "#ff6b6b";
    context.lineWidth = 1.5;
    context.fillRect(x, y, cardWidth, cardHeight);
    context.strokeRect(x, y, cardWidth, cardHeight);
    context.fillStyle = card.valid ? "#f1f5f9" : "#ffb7b7";
    context.font = "bold 13px system-ui";
    context.fillText(card.label, x + 10, y + 25);
    context.fillStyle = "#a7b4c2";
    context.font = "12px ui-monospace, monospace";
    context.fillText(card.shape ? formatShape(card.shape) : "invalid shape", x + 10, y + 49);
    if (card.error) {
      context.fillStyle = "#ff8f8f";
      context.font = "10px system-ui";
      context.fillText(card.error.slice(0, 28), x + 10, y + 72);
    }
    if (index < cards.length - 1 && column < columns - 1) {
      context.strokeStyle = "#39d0c7";
      context.beginPath();
      context.moveTo(x + cardWidth + 2, y + cardHeight / 2);
      context.lineTo(x + cardWidth + 10, y + cardHeight / 2);
      context.stroke();
    }
  });
}

function drawGanDistribution() {
  const { context, width, height } = resizeCanvas(elements.decisionCanvas);
  context.clearRect(0, 0, width, height);
  const padding = 28;
  const plotWidth = Math.max(1, width - padding * 2);
  const plotHeight = Math.max(1, height - padding * 2);
  const toCanvas = ([x, y]) => [
    padding + (x + 1.2) / 2.4 * plotWidth,
    padding + (1.2 - y) / 2.4 * plotHeight,
  ];

  context.strokeStyle = "#2d3d52";
  context.lineWidth = 1;
  for (let value = -1; value <= 1; value += 0.5) {
    const [x] = toCanvas([value, 0]);
    const [, y] = toCanvas([0, value]);
    context.beginPath();
    context.moveTo(x, padding);
    context.lineTo(x, height - padding);
    context.stroke();
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.stroke();
  }

  context.fillStyle = "#57df8b";
  for (let index = 0; index < 120; index += 1) {
    const angle = index / 120 * Math.PI * 2;
    const radius = 0.65 + Math.sin(index * 2.17) * 0.035;
    const [x, y] = toCanvas([Math.cos(angle) * radius, Math.sin(angle) * radius]);
    context.beginPath();
    context.arc(x, y, 2.6, 0, Math.PI * 2);
    context.fill();
  }

  const generated = advancedTrainingSession?.generatedPoints?.(120) ?? [];
  context.fillStyle = "#ff7c78";
  for (const point of generated) {
    const [x, y] = toCanvas(point);
    if (x < padding || x > width - padding || y < padding || y > height - padding) continue;
    context.beginPath();
    context.arc(x, y, 2.3, 0, Math.PI * 2);
    context.fill();
  }

  context.font = "12px system-ui";
  context.fillStyle = "#57df8b";
  context.fillText("target distribution", padding, 18);
  context.fillStyle = "#ff7c78";
  context.fillText("generated samples", padding + 124, 18);
}

function drawAdvancedVisuals() {
  const family = elements.modelFamily.value;
  elements.datasetBadge.textContent = MODEL_FAMILIES[family].label;
  elements.primaryVisualTitle.textContent = family === "gan"
    ? "실시간 생성 분포"
    : "Tensor shape flow";
  elements.primaryVisualDescription.textContent = family === "gan"
    ? "초록색은 목표 데이터, 빨간색은 Generator가 만든 샘플입니다."
    : "각 레이어를 통과하며 tensor shape가 어떻게 바뀌는지 검증합니다.";
  elements.networkVisualTitle.textContent = "TensorFlow.js 모델";
  elements.networkVisualDescription.textContent = "실제로 생성된 모델과 파라미터 정보";
  elements.lossVisualTitle.textContent = "실시간 학습 곡선";
  elements.lossVisualDescription.textContent = family === "gan"
    ? "Generator / Discriminator loss"
    : "Train / validation loss";
  elements.probeValue.textContent = elements.modelFamily.value === "gan"
    ? activeGanBranch
    : formatShape(MODEL_FAMILIES[family].inputShape);
  const trainingMetrics = advancedTrainingSession?.metrics?.();
  elements.epochBadge.textContent = advancedCompileInfo?.error
    ? "Invalid"
    : `${family === "gan" ? "Step" : "Epoch"} ${trainingMetrics?.epoch ?? 0}`;

  if (family === "gan") drawGanDistribution();
  else drawShapePipeline();
  const branchLines = family === "gan"
    ? [
        `Generator: ${compiledAdvancedModels?.generator?.countParams?.() ?? 0} parameters`,
        `Discriminator: ${compiledAdvancedModels?.discriminator?.countParams?.() ?? 0} parameters`,
        `Editing branch: ${activeGanBranch}`,
      ]
    : [
        `${MODEL_FAMILIES[family].label}`,
        `${advancedCompileInfo?.parameterCount ?? 0} trainable parameters`,
        advancedCompileInfo?.output ?? "No compiled output",
      ];
  drawTextPanel(
    elements.networkCanvas,
    advancedCompileInfo?.error ? "Compile failed" : "Model compiled",
    branchLines,
    advancedCompileInfo?.error ? "#ff6b6b" : "#57df8b",
  );
  drawLossChart(advancedTrainingSession?.history ?? []);
  elements.trainLossLabel.textContent = family === "gan" ? "Generator loss" : "Train loss";
  elements.validationLossLabel.textContent =
    family === "gan" ? "Discriminator loss" : "Val loss";
  elements.trainAccuracyLabel.textContent = family === "gan" ? "Train acc (N/A)" : "Train acc";
  elements.validationAccuracyLabel.textContent =
    family === "gan" ? "Val acc (N/A)" : "Val acc";
  const formatMetric = (value, percent = false) => Number.isFinite(value)
    ? percent
      ? `${(value * 100).toFixed(1)}%`
      : value.toFixed(4)
    : "-";
  elements.trainLoss.textContent = formatMetric(trainingMetrics?.trainLoss);
  elements.validationLoss.textContent = formatMetric(trainingMetrics?.validationLoss);
  elements.trainAccuracy.textContent = formatMetric(trainingMetrics?.trainAccuracy, true);
  elements.validationAccuracy.textContent = formatMetric(
    trainingMetrics?.validationAccuracy,
    true,
  );
}

function drawAll() {
  if (isAdvancedFamily()) {
    drawAdvancedVisuals();
    return;
  }
  if (!session) return;
  drawDecisionBoundary();
  drawNetwork();
  drawLossChart();
  updateMetrics();
}

async function runAdvancedTrainingStep(stepCount = Number(elements.trainingSpeed.value)) {
  if (!advancedTrainingSession || advancedTrainingPending) return false;
  const trainingSession = advancedTrainingSession;
  const currentEpoch = trainingSession.metrics().epoch;
  const remainingEpochs = Number(elements.maxEpochs.value) - currentEpoch;
  if (remainingEpochs <= 0) return false;

  trainingSession.updateConfig({
    optimizer: elements.optimizer.value,
    learningRate: Number(elements.learningRate.value),
    batchSize: Number(elements.batchSize.value),
  });
  advancedTrainingPending = true;
  try {
    await trainingSession.step(Math.min(stepCount, remainingEpochs));
    if (advancedTrainingSession === trainingSession) drawAdvancedVisuals();
    return true;
  } catch (error) {
    if (advancedTrainingSession === trainingSession) {
      running = false;
      updateStatus();
      setStatus(`학습 오류: ${error.message}`, false);
    }
    return false;
  } finally {
    if (advancedTrainingSession === trainingSession) {
      advancedTrainingPending = false;
    }
  }
}

function animationFrame(timestamp) {
  if (running) {
    const currentEpoch = isAdvancedFamily()
      ? advancedTrainingSession?.metrics?.().epoch ?? 0
      : session.epoch;
    if (currentEpoch >= Number(elements.maxEpochs.value)) {
      running = false;
      updateStatus();
      setStatus("학습 완료", false);
    } else if (isAdvancedFamily()) {
      if (timestamp - lastAdvancedTrainingAt >= 120) {
        runAdvancedTrainingStep();
        lastAdvancedTrainingAt = timestamp;
      }
    } else {
      session.updateTrainingConfig({
        optimizer: elements.optimizer.value,
        learningRate: Number(elements.learningRate.value),
        batchSize: Number(elements.batchSize.value),
      });
      session.step(Number(elements.trainingSpeed.value));
    }
  }

  if (timestamp - lastVisualUpdate >= 90) {
    drawAll();
    lastVisualUpdate = timestamp;
  }
  requestAnimationFrame(animationFrame);
}

function setupPalette() {
  elements.layerPalette.addEventListener("click", (event) => {
    const block = event.target.closest(".palette-block");
    if (!block) return;
    if (isAdvancedFamily()) addAdvancedLayer(block.dataset.layerType);
    else addLayer(block.dataset.activation);
  });

  elements.layerPalette.addEventListener("dragstart", (event) => {
    const block = event.target.closest(".palette-block");
    if (!block) return;
    if (isAdvancedFamily()) {
      event.dataTransfer.setData("application/x-neural-layer-type", block.dataset.layerType);
    } else {
      event.dataTransfer.setData("application/x-neural-activation", block.dataset.activation);
    }
    event.dataTransfer.effectAllowed = "copy";
  });
}

function setupArchitectureEvents() {
  elements.networkStack.addEventListener("click", (event) => {
    const block = event.target.closest(isAdvancedFamily() ? ".advanced-block" : ".hidden-block");
    if (!block) return;
    const layerId = block.dataset.layerId;
    if (isAdvancedFamily()) {
      if (event.target.closest(".remove-layer")) removeAdvancedLayer(layerId);
      if (event.target.closest(".move-up")) moveAdvancedLayer(layerId, -1);
      if (event.target.closest(".move-down")) moveAdvancedLayer(layerId, 1);
    } else {
      if (event.target.closest(".remove-layer")) removeLayer(layerId);
      if (event.target.closest(".move-up")) moveLayer(layerId, -1);
      if (event.target.closest(".move-down")) moveLayer(layerId, 1);
    }
  });

  elements.networkStack.addEventListener("change", (event) => {
    const block = event.target.closest(isAdvancedFamily() ? ".advanced-block" : ".hidden-block");
    if (!block) return;
    if (isAdvancedFamily()) {
      const layer = activeAdvancedLayers().find((item) => item.id === block.dataset.layerId);
      if (!layer) return;
      const fieldName = event.target.dataset.field;
      const field = LAYER_CATALOG[layer.type].fields[fieldName];
      if (!field) return;
      if (field.type === "number") layer[fieldName] = Number(event.target.value);
      else if (field.type === "boolean") layer[fieldName] = event.target.value === "true";
      else layer[fieldName] = event.target.value;
      rebuildAdvancedStudio();
    } else {
      const layer = hiddenLayers.find((item) => item.id === block.dataset.layerId);
      if (!layer) return;
      if (event.target.matches(".units-input")) {
        layer.units = Math.max(1, Math.min(32, Number(event.target.value) || 1));
      }
      if (event.target.matches(".activation-select")) {
        layer.activation = event.target.value;
      }
      rebuildModel();
    }
  });

  elements.networkStack.addEventListener("dragstart", (event) => {
    const block = event.target.closest(isAdvancedFamily() ? ".advanced-block" : ".hidden-block");
    if (!block) return;
    draggedLayerId = block.dataset.layerId;
    event.dataTransfer.setData("application/x-neural-layer", draggedLayerId);
    event.dataTransfer.effectAllowed = "move";
  });

  elements.networkStack.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.networkStack.classList.add("drag-over");
  });

  elements.networkStack.addEventListener("dragleave", (event) => {
    if (!elements.networkStack.contains(event.relatedTarget)) {
      elements.networkStack.classList.remove("drag-over");
    }
  });

  elements.networkStack.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.networkStack.classList.remove("drag-over");
    const targetBlock = event.target.closest(isAdvancedFamily() ? ".advanced-block" : ".hidden-block");
    const activation = event.dataTransfer.getData("application/x-neural-activation");
    const layerType = event.dataTransfer.getData("application/x-neural-layer-type");
    const layerId = event.dataTransfer.getData("application/x-neural-layer") || draggedLayerId;
    if (isAdvancedFamily() && layerType) {
      const targetIndex = targetBlock
        ? activeAdvancedLayers().findIndex((layer) => layer.id === targetBlock.dataset.layerId)
        : activeAdvancedLayers().length;
      addAdvancedLayer(layerType, targetIndex);
    } else if (isAdvancedFamily() && layerId && targetBlock) {
      moveAdvancedLayerTo(layerId, targetBlock.dataset.layerId);
    } else if (activation) {
      const targetIndex = targetBlock
        ? hiddenLayers.findIndex((layer) => layer.id === targetBlock.dataset.layerId)
        : hiddenLayers.length;
      addLayer(activation, targetIndex);
    } else if (layerId && targetBlock) {
      moveLayerTo(layerId, targetBlock.dataset.layerId);
    }
    draggedLayerId = null;
  });
}

function setupControls() {
  elements.modelFamily.addEventListener("change", () => {
    running = false;
    activeGanBranch = "generator";
    elements.learningRate.value = String(
      familyLearningRates[elements.modelFamily.value],
    );
    renderPalette();
    if (isAdvancedFamily()) rebuildAdvancedStudio();
    else rebuildModel();
  });

  elements.branchTabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-branch]");
    if (!button) return;
    activeGanBranch = button.dataset.branch;
    rebuildAdvancedStudio();
  });

  elements.toggleTraining.addEventListener("click", () => {
    const currentEpoch = isAdvancedFamily()
      ? advancedTrainingSession?.metrics().epoch ?? 0
      : session.epoch;
    if (!running && currentEpoch >= Number(elements.maxEpochs.value)) {
      if (isAdvancedFamily()) rebuildAdvancedStudio();
      else rebuildModel();
    }
    running = !running;
    updateStatus();
  });

  elements.stepTraining.addEventListener("click", async () => {
    running = false;
    updateStatus();
    if (isAdvancedFamily()) {
      await runAdvancedTrainingStep(1);
    } else {
      session.step(1);
    }
    drawAll();
  });

  elements.resetTraining.addEventListener("click", () => {
    running = false;
    if (isAdvancedFamily()) {
      rebuildAdvancedStudio();
    } else {
      rebuildModel();
    }
  });

  for (const control of [elements.datasetType, elements.dataCount, elements.noise]) {
    control.addEventListener("input", () => {
      running = false;
      rebuildModel();
    });
  }

  elements.optimizer.addEventListener("change", () => {
    if (isAdvancedFamily()) {
      if (!advancedTrainingPending) {
        advancedTrainingSession?.updateConfig({ optimizer: elements.optimizer.value });
      }
    } else {
      session.updateTrainingConfig({ optimizer: elements.optimizer.value });
      session.network.resetOptimizerState();
    }
  });

  for (const control of [
    elements.learningRate,
    elements.batchSize,
    elements.trainingSpeed,
    elements.maxEpochs,
  ]) {
    control.addEventListener("input", () => {
      updateControlLabels();
      if (isAdvancedFamily()) {
        if (!advancedTrainingPending) {
          advancedTrainingSession?.updateConfig({
            learningRate: Number(elements.learningRate.value),
            batchSize: Number(elements.batchSize.value),
          });
        }
      } else {
        session.updateTrainingConfig({
          learningRate: Number(elements.learningRate.value),
          batchSize: Number(elements.batchSize.value),
        });
      }
    });
  }

  elements.decisionCanvas.addEventListener("click", (event) => {
    if (isAdvancedFamily()) return;
    const rect = elements.decisionCanvas.getBoundingClientRect();
    probe = [
      -1.2 + (event.clientX - rect.left) / rect.width * 2.4,
      1.2 - (event.clientY - rect.top) / rect.height * 2.4,
    ];
    elements.probeValue.textContent = `x: ${probe[0].toFixed(2)}, y: ${probe[1].toFixed(2)}`;
    drawDecisionBoundary();
    drawNetwork();
  });

  window.addEventListener("resize", drawAll);
}

setupPalette();
setupArchitectureEvents();
setupControls();
renderPalette();
updateControlLabels();
rebuildModel();
requestAnimationFrame(animationFrame);
