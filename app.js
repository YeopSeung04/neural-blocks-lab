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
import {
  parseCsvText,
  prepareImageDataset,
  prepareTabularDataset,
  prepareTimeSeriesDataset,
  suggestColumnMapping,
  summarizeImageFolder,
} from "./data-pipeline.mjs";

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
  dataSource: document.getElementById("dataSource"),
  dataPrivacyBadge: document.getElementById("dataPrivacyBadge"),
  uploadControls: document.getElementById("uploadControls"),
  fileInputRow: document.getElementById("fileInputRow"),
  folderInputRow: document.getElementById("folderInputRow"),
  dataFileInput: document.getElementById("dataFileInput"),
  imageFolderInput: document.getElementById("imageFolderInput"),
  columnMapping: document.getElementById("columnMapping"),
  feature1Row: document.getElementById("feature1Row"),
  feature2Row: document.getElementById("feature2Row"),
  featureColumn1: document.getElementById("featureColumn1"),
  featureColumn2: document.getElementById("featureColumn2"),
  targetColumn: document.getElementById("targetColumn"),
  preprocessPipeline: document.getElementById("preprocessPipeline"),
  applyPreprocessing: document.getElementById("applyPreprocessing"),
  dataSummary: document.getElementById("dataSummary"),
  builtInDataControls: document.getElementById("builtInDataControls"),
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
  weightTooltip: document.getElementById("weightTooltip"),
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
  resourceStatus: document.getElementById("resourceStatus"),
  cpuValue: document.getElementById("cpuValue"),
  cpuBar: document.getElementById("cpuBar"),
  cpuSpark: document.getElementById("cpuSpark"),
  cpuDetail: document.getElementById("cpuDetail"),
  ramValue: document.getElementById("ramValue"),
  ramBar: document.getElementById("ramBar"),
  ramSpark: document.getElementById("ramSpark"),
  ramDetail: document.getElementById("ramDetail"),
  gpuValue: document.getElementById("gpuValue"),
  gpuBar: document.getElementById("gpuBar"),
  gpuSpark: document.getElementById("gpuSpark"),
  gpuDetail: document.getElementById("gpuDetail"),
  vramValue: document.getElementById("vramValue"),
  vramBar: document.getElementById("vramBar"),
  vramSpark: document.getElementById("vramSpark"),
  vramDetail: document.getElementById("vramDetail"),
  tfBackendValue: document.getElementById("tfBackendValue"),
  tfTensorCount: document.getElementById("tfTensorCount"),
  tfMemoryValue: document.getElementById("tfMemoryValue"),
  jsHeapValue: document.getElementById("jsHeapValue"),
  tfDetail: document.getElementById("tfDetail"),
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
let parsedUpload = null;
let selectedImageFiles = [];
let activeUserDataset = null;
let weightHitTargets = [];
let hoveredWeightKey = null;
let weightPointer = null;
let lastFrameTimestamp = performance.now();
let mainThreadLoad = 0;
const preprocessing = {
  missingStrategy: "drop",
  scaling: "standard",
  validationRatio: 0.2,
  imageSize: 8,
  pixelScaling: "zeroOne",
  sequenceLength: 12,
  stride: 1,
};
const resourceHistory = {
  cpu: [],
  ram: [],
  gpu: [],
  vram: [],
};
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
    points: activeUserDataset?.family === "mlp" ? activeUserDataset.points : null,
    validationRatio: preprocessing.validationRatio,
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
  if (activeUserDataset?.family === family) return activeUserDataset.inputShape;
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
    validationRatio: preprocessing.validationRatio,
  };
  advancedTrainingSession = family === "gan"
    ? new GanTrainingSession(tf, advancedArchitectures.gan, {
        ...config,
        models: compiledAdvancedModels,
      })
    : new TfClassifierSession(tf, family, activeAdvancedLayers(), {
        ...config,
        model: compiledAdvancedModels,
        datasetData: activeUserDataset?.family === family
          ? activeUserDataset.datasetData
          : null,
        inputShape: advancedInputShape(),
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
      compiledAdvancedModels = buildClassifier(tf, family, layers, advancedInputShape());
      advancedCompileInfo.parameterCount = compiledAdvancedModels.countParams();
      advancedCompileInfo.output =
        `[batch, ${advancedInputShape().join(", ")}] -> [batch, 1]`;
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

function usesActiveUserDataset() {
  return activeUserDataset?.family === elements.modelFamily.value;
}

function activeDatasetName() {
  if (usesActiveUserDataset()) return activeUserDataset.name;
  if (isAdvancedFamily()) {
    return {
      cnn: "8x8 선 방향 이미지",
      rnn: "12-step 상승/하락 시계열",
      gan: "2D 원형 목표 분포",
    }[elements.modelFamily.value];
  }
  return datasetLabels[elements.datasetType.value];
}

function updateControlLabels() {
  const uploadedCount = usesActiveUserDataset()
    ? activeUserDataset.points?.length ?? activeUserDataset.datasetData?.count
    : null;
  elements.dataCountValue.textContent = uploadedCount ?? elements.dataCount.value;
  elements.noiseValue.textContent = Number(elements.noise.value).toFixed(2);
  elements.learningRateValue.textContent = Number(elements.learningRate.value).toFixed(3);
  elements.batchSizeValue.textContent = elements.batchSize.value;
  elements.speedValue.textContent = elements.trainingSpeed.value;
  elements.maxEpochsValue.textContent = elements.maxEpochs.value;
  elements.datasetBadge.textContent = activeDatasetName();
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
  elements.networkVisualDescription.textContent =
    "가중치 선에 마우스를 올리면 현재 forward 계산식을 확인할 수 있습니다.";
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
  const uploaded = usesActiveUserDataset();
  elements.datasetType.disabled = advanced || uploaded;
  elements.noise.disabled = advanced || uploaded;
  elements.dataCount.disabled = uploaded || (advanced && family === "gan");
  datasetOptions.forEach((option, index) => {
    option.hidden = advanced && index > 0;
  });
  datasetOptions[0].textContent = uploaded
    ? activeUserDataset.name
    : advanced
      ? {
          cnn: "8x8 선 방향 이미지",
          rnn: "12-step 상승/하락 시계열",
          gan: "2D 원형 목표 분포",
        }[family]
      : "XOR";
  if (advanced || uploaded) elements.datasetType.selectedIndex = 0;
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

function setDataSummary(text, state = "neutral") {
  elements.dataSummary.className = "data-summary";
  if (state === "valid" || state === "invalid") {
    elements.dataSummary.classList.add(state);
  }
  elements.dataSummary.textContent = text;
}

function createPreprocessBlock(title, description, setting, options, value) {
  const block = document.createElement("div");
  block.className = "preprocess-block";
  const text = document.createElement("div");
  const heading = document.createElement("strong");
  heading.textContent = title;
  const detail = document.createElement("small");
  detail.textContent = description;
  text.append(heading, detail);

  const select = document.createElement("select");
  select.dataset.preprocessSetting = setting;
  for (const optionDefinition of options) {
    const option = document.createElement("option");
    const [optionValue, label] = Array.isArray(optionDefinition)
      ? optionDefinition
      : [optionDefinition, optionDefinition];
    option.value = String(optionValue);
    option.textContent = label;
    select.append(option);
  }
  select.value = String(value);
  block.append(text, select);
  return block;
}

function renderPreprocessingPipeline() {
  const source = elements.dataSource.value;
  elements.preprocessPipeline.replaceChildren();
  if (source === "builtIn") return;

  if (source !== "imageFolder") {
    elements.preprocessPipeline.append(createPreprocessBlock(
      "결측치 처리",
      "빈 숫자 셀을 제거하거나 열 평균으로 대체",
      "missingStrategy",
      [["drop", "행 제거"], ["mean", "평균값 대체"]],
      preprocessing.missingStrategy,
    ));
    elements.preprocessPipeline.append(createPreprocessBlock(
      "Feature scaling",
      "학습 전에 숫자 feature 범위를 조정",
      "scaling",
      [["standard", "Standard (z-score)"], ["minmax", "MinMax (-1~1)"], ["none", "변환 없음"]],
      preprocessing.scaling,
    ));
  }

  if (source === "imageFolder") {
    elements.preprocessPipeline.append(createPreprocessBlock(
      "Resize + Grayscale",
      "모든 이미지를 같은 크기의 1채널 tensor로 변환",
      "imageSize",
      [[8, "8 x 8"], [16, "16 x 16"], [28, "28 x 28"], [32, "32 x 32"]],
      preprocessing.imageSize,
    ));
    elements.preprocessPipeline.append(createPreprocessBlock(
      "Pixel normalization",
      "픽셀 범위를 CNN 입력에 맞게 변환",
      "pixelScaling",
      [["zeroOne", "0 ~ 1"], ["minusOneOne", "-1 ~ 1"]],
      preprocessing.pixelScaling,
    ));
  }

  if (source === "timeSeries") {
    elements.preprocessPipeline.append(createPreprocessBlock(
      "Sequence window",
      "연속 행을 하나의 RNN 입력 시퀀스로 묶음",
      "sequenceLength",
      [[8, "8 steps"], [12, "12 steps"], [16, "16 steps"], [24, "24 steps"]],
      preprocessing.sequenceLength,
    ));
    elements.preprocessPipeline.append(createPreprocessBlock(
      "Window stride",
      "다음 시퀀스가 시작되는 행 간격",
      "stride",
      [[1, "1 row"], [2, "2 rows"], [4, "4 rows"], [8, "8 rows"]],
      preprocessing.stride,
    ));
  }

  elements.preprocessPipeline.append(createPreprocessBlock(
    "Train / validation split",
    "validation 데이터는 가중치 업데이트에 사용하지 않음",
    "validationRatio",
    [[0.1, "90 / 10"], [0.2, "80 / 20"], [0.25, "75 / 25"], [0.3, "70 / 30"], [0.4, "60 / 40"]],
    preprocessing.validationRatio,
  ));
}

function populateColumnSelect(select, headers, selectedValue) {
  select.replaceChildren();
  for (const header of headers) {
    const option = document.createElement("option");
    option.value = header;
    option.textContent = header;
    select.append(option);
  }
  if (headers.includes(selectedValue)) select.value = selectedValue;
}

function updateColumnMapping(parsed) {
  const mapping = suggestColumnMapping(parsed.headers);
  populateColumnSelect(elements.featureColumn1, parsed.headers, mapping.signalColumn);
  populateColumnSelect(
    elements.featureColumn2,
    parsed.headers,
    mapping.featureColumns[1] ?? mapping.featureColumns[0],
  );
  populateColumnSelect(elements.targetColumn, parsed.headers, mapping.targetColumn);
  elements.columnMapping.hidden = false;
}

function renderDataSourceControls() {
  const source = elements.dataSource.value;
  const builtIn = source === "builtIn";
  elements.uploadControls.hidden = builtIn;
  elements.builtInDataControls.hidden = !builtIn;
  elements.fileInputRow.hidden = builtIn || source === "imageFolder";
  elements.folderInputRow.hidden = source !== "imageFolder";
  elements.columnMapping.hidden =
    source === "imageFolder" || !parsedUpload;
  elements.feature2Row.hidden = source === "timeSeries";
  elements.dataPrivacyBadge.textContent = builtIn ? "내장 데이터" : "로컬 브라우저 처리";
  elements.dataFileInput.accept = source === "timeSeries"
    ? ".csv,.tsv,text/csv,text/tab-separated-values"
    : ".csv,.tsv,text/csv,text/tab-separated-values";
  renderPreprocessingPipeline();
}

function resetUploadedData() {
  parsedUpload = null;
  selectedImageFiles = [];
  activeUserDataset = null;
  elements.dataFileInput.value = "";
  elements.imageFolderInput.value = "";
  elements.applyPreprocessing.disabled = true;
  setDataSummary("파일을 선택하세요.");
}

function describePreparedDataset(dataset) {
  if (dataset.family === "mlp") {
    return [
      `전처리 완료: ${dataset.points.length} rows`,
      `features: ${dataset.summary.features.join(", ")}`,
      `classes: ${dataset.summary.classes.join(" / ")}`,
      `scaling: ${dataset.summary.scaling}`,
    ].join("\n");
  }
  if (dataset.family === "cnn") {
    return [
      `전처리 완료: ${dataset.datasetData.count} images`,
      `shape: [${dataset.inputShape.join(", ")}]`,
      `classes: ${dataset.summary.classes.join(" / ")}`,
      `pixels: ${dataset.summary.pixelScaling}`,
    ].join("\n");
  }
  return [
    `전처리 완료: ${dataset.datasetData.count} windows`,
    `shape: [${dataset.inputShape.join(", ")}]`,
    `signal: ${dataset.summary.signal}`,
    `classes: ${dataset.summary.classes.join(" / ")}`,
  ].join("\n");
}

function rebuildForCurrentFamily() {
  if (isAdvancedFamily()) rebuildAdvancedStudio();
  else rebuildModel();
}

async function applyUploadedDataset() {
  const source = elements.dataSource.value;
  elements.applyPreprocessing.disabled = true;
  setDataSummary("전처리 중...");
  running = false;
  try {
    let prepared;
    let name;
    if (source === "csv") {
      prepared = prepareTabularDataset(parsedUpload, {
        featureColumns: [
          elements.featureColumn1.value,
          elements.featureColumn2.value,
        ],
        targetColumn: elements.targetColumn.value,
        missingStrategy: preprocessing.missingStrategy,
        scaling: preprocessing.scaling,
      });
      name = elements.dataFileInput.files[0]?.name ?? "uploaded.csv";
    } else if (source === "timeSeries") {
      prepared = prepareTimeSeriesDataset(parsedUpload, {
        signalColumn: elements.featureColumn1.value,
        targetColumn: elements.targetColumn.value,
        sequenceLength: preprocessing.sequenceLength,
        stride: preprocessing.stride,
        missingStrategy: preprocessing.missingStrategy,
        scaling: preprocessing.scaling,
      });
      name = elements.dataFileInput.files[0]?.name ?? "timeseries.csv";
    } else if (source === "imageFolder") {
      prepared = await prepareImageDataset(selectedImageFiles, {
        imageSize: preprocessing.imageSize,
        pixelScaling: preprocessing.pixelScaling,
      });
      name = selectedImageFiles[0]?.webkitRelativePath?.split("/")[0] ?? "image folder";
    } else {
      throw new Error("Select an upload data source first");
    }
    activeUserDataset = {
      ...prepared,
      name,
      source,
    };
    if (elements.modelFamily.value !== prepared.family) {
      elements.modelFamily.value = prepared.family;
      elements.modelFamily.dispatchEvent(new Event("change"));
    } else {
      rebuildForCurrentFamily();
    }
    setDataSummary(describePreparedDataset(activeUserDataset), "valid");
  } catch (error) {
    activeUserDataset = null;
    setDataSummary(`전처리 오류: ${error.message}`, "invalid");
    rebuildForCurrentFamily();
  } finally {
    elements.applyPreprocessing.disabled =
      source === "imageFolder" ? !selectedImageFiles.length : !parsedUpload;
  }
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

function distanceToSegment(pointX, pointY, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(pointX - x1, pointY - y1);
  const projection = Math.max(
    0,
    Math.min(1, ((pointX - x1) * dx + (pointY - y1) * dy) / lengthSquared),
  );
  const closestX = x1 + projection * dx;
  const closestY = y1 + projection * dy;
  return Math.hypot(pointX - closestX, pointY - closestY);
}

function updateWeightTooltip(target) {
  if (!target || !weightPointer || isAdvancedFamily()) {
    elements.weightTooltip.hidden = true;
    return;
  }
  const activationName = target.activation.toUpperCase();
  elements.weightTooltip.innerHTML =
    `<strong>Layer ${target.layerIndex + 1} · Weight ${target.unitIndex + 1},${target.inputIndex + 1}</strong><br>` +
    `a = ${target.inputValue.toFixed(5)}<br>` +
    `w = ${target.weight.toFixed(5)}<br>` +
    `a × w = <b>${target.contribution.toFixed(5)}</b><br>` +
    `Σ(a × w) + bias(${target.bias.toFixed(5)}) = z(${target.z.toFixed(5)})<br>` +
    `${activationName}(z) = <b>${target.output.toFixed(5)}</b>`;
  elements.weightTooltip.hidden = false;
  const parent = elements.weightTooltip.parentElement;
  const maxLeft = Math.max(8, parent.clientWidth - elements.weightTooltip.offsetWidth - 8);
  const maxTop = Math.max(8, parent.clientHeight - elements.weightTooltip.offsetHeight - 8);
  elements.weightTooltip.style.left = `${Math.max(8, Math.min(maxLeft, weightPointer.left))}px`;
  elements.weightTooltip.style.top = `${Math.max(8, Math.min(maxTop, weightPointer.top))}px`;
}

function drawNetwork() {
  const { context, width, height } = resizeCanvas(elements.networkCanvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f151c";
  context.fillRect(0, 0, width, height);

  const architecture = session.network.architecture();
  const snapshot = session.network.activationSnapshot(probe);
  weightHitTargets = [];
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
        const x1 = previous.x;
        const y1 = nodeY(previousDisplay, previous.indices.length);
        const x2 = current.x;
        const y2 = nodeY(currentDisplay, current.indices.length);
        const key = `${layerIndex - 1}:${currentIndex}:${previousIndex}`;
        const hovered = hoveredWeightKey === key;
        context.strokeStyle = hovered
          ? "#f7d66d"
          : weight >= 0
            ? `rgba(57, 208, 199, ${0.12 + magnitude * 0.68})`
            : `rgba(255, 107, 107, ${0.12 + magnitude * 0.68})`;
        context.lineWidth = hovered ? 4 : 0.5 + magnitude * 2.2;
        context.beginPath();
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
        context.stroke();
        const detail = snapshot.details[layerIndex - 1];
        const inputValue = previous.values[previousIndex] ?? 0;
        weightHitTargets.push({
          key,
          x1,
          y1,
          x2,
          y2,
          layerIndex: layerIndex - 1,
          unitIndex: currentIndex,
          inputIndex: previousIndex,
          inputValue,
          weight,
          contribution: inputValue * weight,
          bias: weightLayer.biases[currentIndex],
          z: detail.z[currentIndex],
          output: detail.output[currentIndex],
          activation: weightLayer.activation,
        });
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
  updateWeightTooltip(
    weightHitTargets.find((target) => target.key === hoveredWeightKey) ?? null,
  );
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
  elements.weightTooltip.hidden = true;
  elements.datasetBadge.textContent = usesActiveUserDataset()
    ? activeUserDataset.name
    : MODEL_FAMILIES[family].label;
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

function formatBytes(value) {
  if (!Number.isFinite(value)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = Math.max(0, value);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function appendResourceHistory(key, value) {
  if (!Number.isFinite(value)) return;
  resourceHistory[key].push(Math.max(0, Math.min(100, value)));
  if (resourceHistory[key].length > 48) resourceHistory[key].shift();
}

function drawResourceSpark(canvas, values, color = "#39d0c7") {
  const { context, width, height } = resizeCanvas(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f151c";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(255,255,255,0.08)";
  context.lineWidth = 1;
  for (let index = 1; index < 4; index += 1) {
    const y = height * index / 4;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  if (!values.length) return;
  context.strokeStyle = color;
  context.lineWidth = 1.8;
  context.beginPath();
  values.forEach((value, index) => {
    const x = values.length === 1 ? width : index / (values.length - 1) * width;
    const y = height - value / 100 * height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function setResourceBar(bar, value) {
  bar.style.width = Number.isFinite(value)
    ? `${Math.max(0, Math.min(100, value))}%`
    : "0%";
}

function updateTensorFlowMetrics() {
  const tf = globalThis.tf;
  if (!tf) {
    elements.tfBackendValue.textContent = "unavailable";
    return;
  }
  const memory = tf.memory();
  elements.tfBackendValue.textContent = tf.getBackend();
  elements.tfTensorCount.textContent = memory.numTensors.toLocaleString();
  elements.tfMemoryValue.textContent = formatBytes(memory.numBytes);
  const heap = performance.memory;
  elements.jsHeapValue.textContent = heap
    ? `${formatBytes(heap.usedJSHeapSize)} / ${formatBytes(heap.jsHeapSizeLimit)}`
    : "browser 제한";
  elements.tfDetail.textContent = memory.unreliable
    ? `Tensor bytes는 ${memory.reasons?.join(", ") || "backend 특성"}으로 추정값`
    : "현재 페이지가 보유한 TensorFlow.js tensor 메모리";
}

async function updateResourceMonitor() {
  updateTensorFlowMetrics();
  try {
    const response = await fetch("/api/system-metrics", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const metrics = await response.json();
    const cpu = metrics.cpu.usagePercent;
    const ram = metrics.memory.usagePercent;
    const gpu = metrics.gpu.usagePercent;
    const gpuMemoryUsed =
      metrics.gpu.memoryUsedBytes ?? metrics.gpu.memoryAllocatedBytes;
    const gpuMemoryTotal = metrics.gpu.memoryTotalBytes;
    const vram = Number.isFinite(gpuMemoryUsed) && Number.isFinite(gpuMemoryTotal)
      ? gpuMemoryUsed / gpuMemoryTotal * 100
      : NaN;

    elements.resourceStatus.textContent = "OS bridge 연결";
    elements.cpuValue.textContent = Number.isFinite(cpu) ? `${cpu.toFixed(1)}%` : "-";
    elements.cpuDetail.textContent =
      `${metrics.cpu.logicalCores ?? navigator.hardwareConcurrency ?? "-"} logical cores · ${metrics.cpu.source}`;
    elements.ramValue.textContent = Number.isFinite(ram) ? `${ram.toFixed(1)}%` : "-";
    elements.ramDetail.textContent =
      `${formatBytes(metrics.memory.usedBytes)} / ${formatBytes(metrics.memory.totalBytes)}`;
    elements.gpuValue.textContent = Number.isFinite(gpu) ? `${gpu.toFixed(1)}%` : "N/A";
    elements.gpuDetail.textContent =
      `${metrics.gpu.name ?? "GPU"} · ${metrics.gpu.source}`;
    elements.vramValue.textContent = Number.isFinite(vram) ? `${vram.toFixed(1)}%` : "N/A";
    elements.vramDetail.textContent = Number.isFinite(gpuMemoryUsed)
      ? `${formatBytes(gpuMemoryUsed)} used · ${metrics.gpu.memoryType}`
      : `실제 VRAM 측정 불가 · ${metrics.gpu.memoryType}`;

    setResourceBar(elements.cpuBar, cpu);
    setResourceBar(elements.ramBar, ram);
    setResourceBar(elements.gpuBar, gpu);
    setResourceBar(elements.vramBar, vram);
    appendResourceHistory("cpu", cpu);
    appendResourceHistory("ram", ram);
    appendResourceHistory("gpu", gpu);
    appendResourceHistory("vram", vram);
  } catch {
    const heap = performance.memory;
    const heapPercent = heap
      ? heap.usedJSHeapSize / heap.jsHeapSizeLimit * 100
      : NaN;
    elements.resourceStatus.textContent = "브라우저 제한 모드";
    elements.cpuValue.textContent = `${mainThreadLoad.toFixed(1)}%`;
    elements.cpuDetail.textContent =
      `UI thread 부하 추정 · ${navigator.hardwareConcurrency ?? "-"} logical cores`;
    elements.ramValue.textContent = Number.isFinite(heapPercent)
      ? `${heapPercent.toFixed(1)}%`
      : "N/A";
    elements.ramDetail.textContent = heap
      ? `JS heap ${formatBytes(heap.usedJSHeapSize)} / ${formatBytes(heap.jsHeapSizeLimit)}`
      : `deviceMemory 약 ${navigator.deviceMemory ?? "-"} GB · 실제 사용률 제한`;
    elements.gpuValue.textContent = "N/A";
    elements.gpuDetail.textContent = `${globalThis.tf?.getBackend?.() ?? "GPU"} backend · 사용률 API 제한`;
    elements.vramValue.textContent = "N/A";
    elements.vramDetail.textContent = "브라우저는 실제 VRAM 사용량을 제공하지 않음";
    setResourceBar(elements.cpuBar, mainThreadLoad);
    setResourceBar(elements.ramBar, heapPercent);
    setResourceBar(elements.gpuBar, NaN);
    setResourceBar(elements.vramBar, NaN);
    appendResourceHistory("cpu", mainThreadLoad);
    appendResourceHistory("ram", heapPercent);
  }

  drawResourceSpark(elements.cpuSpark, resourceHistory.cpu);
  drawResourceSpark(elements.ramSpark, resourceHistory.ram, "#67a9ff");
  drawResourceSpark(elements.gpuSpark, resourceHistory.gpu, "#f7d66d");
  drawResourceSpark(elements.vramSpark, resourceHistory.vram, "#ffad5a");
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
  const frameDuration = Math.max(0, timestamp - lastFrameTimestamp);
  lastFrameTimestamp = timestamp;
  const frameLoad = Math.max(0, Math.min(100, (frameDuration - 16.7) / 50 * 100));
  mainThreadLoad = mainThreadLoad * 0.9 + frameLoad * 0.1;
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

function setupDataControls() {
  elements.dataSource.addEventListener("change", () => {
    running = false;
    resetUploadedData();
    renderDataSourceControls();
    rebuildForCurrentFamily();
  });

  elements.dataFileInput.addEventListener("change", async () => {
    const file = elements.dataFileInput.files[0];
    if (!file) {
      parsedUpload = null;
      elements.applyPreprocessing.disabled = true;
      setDataSummary("파일을 선택하세요.");
      return;
    }
    try {
      parsedUpload = parseCsvText(await file.text());
      updateColumnMapping(parsedUpload);
      renderDataSourceControls();
      elements.applyPreprocessing.disabled = false;
      setDataSummary(
        `${file.name}\n${parsedUpload.rows.length} rows · ${parsedUpload.headers.length} columns`,
      );
    } catch (error) {
      parsedUpload = null;
      elements.applyPreprocessing.disabled = true;
      setDataSummary(`파일 오류: ${error.message}`, "invalid");
    }
  });

  elements.imageFolderInput.addEventListener("change", () => {
    selectedImageFiles = [...elements.imageFolderInput.files];
    try {
      const summary = summarizeImageFolder(selectedImageFiles);
      elements.applyPreprocessing.disabled = summary.imageCount < 1;
      setDataSummary(
        `${summary.imageCount} images\nclasses: ${summary.classes.join(" / ") || "확인 불가"}`,
      );
    } catch (error) {
      selectedImageFiles = [];
      elements.applyPreprocessing.disabled = true;
      setDataSummary(`폴더 오류: ${error.message}`, "invalid");
    }
  });

  elements.preprocessPipeline.addEventListener("change", (event) => {
    const setting = event.target.dataset.preprocessSetting;
    if (!setting) return;
    const numericSettings = new Set([
      "validationRatio",
      "imageSize",
      "sequenceLength",
      "stride",
    ]);
    preprocessing[setting] = numericSettings.has(setting)
      ? Number(event.target.value)
      : event.target.value;
    elements.applyPreprocessing.disabled =
      elements.dataSource.value === "imageFolder"
        ? !selectedImageFiles.length
        : !parsedUpload;
    setDataSummary("전처리 설정이 변경되었습니다. 다시 적용하세요.");
  });

  elements.columnMapping.addEventListener("change", () => {
    elements.applyPreprocessing.disabled = !parsedUpload;
    setDataSummary("컬럼 매핑이 변경되었습니다. 전처리를 적용하세요.");
  });

  elements.applyPreprocessing.addEventListener("click", applyUploadedDataset);
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

  elements.networkCanvas.addEventListener("mousemove", (event) => {
    if (isAdvancedFamily()) {
      hoveredWeightKey = null;
      elements.weightTooltip.hidden = true;
      return;
    }
    const rect = elements.networkCanvas.getBoundingClientRect();
    const pointX = event.clientX - rect.left;
    const pointY = event.clientY - rect.top;
    let nearest = null;
    let nearestDistance = 7;
    for (const target of weightHitTargets) {
      const distance = distanceToSegment(
        pointX,
        pointY,
        target.x1,
        target.y1,
        target.x2,
        target.y2,
      );
      if (distance < nearestDistance) {
        nearest = target;
        nearestDistance = distance;
      }
    }
    const parentRect = elements.weightTooltip.parentElement.getBoundingClientRect();
    weightPointer = {
      left: event.clientX - parentRect.left + 12,
      top: event.clientY - parentRect.top + 12,
    };
    hoveredWeightKey = nearest?.key ?? null;
    drawNetwork();
  });

  elements.networkCanvas.addEventListener("mouseleave", () => {
    hoveredWeightKey = null;
    weightPointer = null;
    elements.weightTooltip.hidden = true;
    if (!isAdvancedFamily()) drawNetwork();
  });

  window.addEventListener("resize", drawAll);
}

setupPalette();
setupArchitectureEvents();
setupDataControls();
setupControls();
renderDataSourceControls();
renderPalette();
updateControlLabels();
rebuildModel();
updateResourceMonitor();
setInterval(updateResourceMonitor, 1500);
requestAnimationFrame(animationFrame);
