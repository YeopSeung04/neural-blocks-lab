export const ACTIVATIONS = [
  { value: "relu", label: "ReLU" },
  { value: "leakyRelu", label: "Leaky ReLU" },
  { value: "elu", label: "ELU" },
  { value: "selu", label: "SELU" },
  { value: "tanh", label: "Tanh" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "softmax", label: "Softmax" },
  { value: "linear", label: "Linear" },
  { value: "swish", label: "Swish" },
  { value: "gelu", label: "GELU" },
  { value: "hardSigmoid", label: "Hard Sigmoid" },
  { value: "softplus", label: "Softplus" },
  { value: "softsign", label: "Softsign" },
];

export const MODEL_FAMILIES = {
  mlp: {
    label: "MLP / Dense",
    description: "표 형태와 2D 좌표 데이터를 처리하는 완전연결 신경망",
    inputShape: [2],
    palette: ["dense", "dropout", "batchNormalization", "activation"],
    template: [
      { type: "dense", units: 8, activation: "tanh" },
    ],
  },
  cnn: {
    label: "CNN",
    description: "이미지의 지역 패턴을 합성곱 필터로 학습하는 신경망",
    inputShape: [8, 8, 1],
    palette: [
      "conv2d",
      "maxPooling2d",
      "averagePooling2d",
      "batchNormalization",
      "dropout",
      "flatten",
      "dense",
      "activation",
    ],
    template: [
      { type: "conv2d", filters: 8, kernelSize: 3, strides: 1, padding: "same", activation: "relu" },
      { type: "maxPooling2d", poolSize: 2, strides: 2, padding: "valid" },
      { type: "conv2d", filters: 12, kernelSize: 3, strides: 1, padding: "same", activation: "relu" },
      { type: "flatten" },
      { type: "dense", units: 12, activation: "relu" },
    ],
  },
  rnn: {
    label: "RNN / Sequence",
    description: "시간 순서가 있는 시퀀스 데이터를 recurrent state로 학습하는 신경망",
    inputShape: [12, 1],
    palette: [
      "simpleRnn",
      "lstm",
      "gru",
      "dropout",
      "batchNormalization",
      "dense",
      "activation",
    ],
    template: [
      { type: "lstm", units: 12, activation: "tanh", recurrentActivation: "sigmoid", returnSequences: false },
      { type: "dense", units: 8, activation: "relu" },
    ],
  },
  gan: {
    label: "GAN",
    description: "Generator와 Discriminator가 서로 경쟁하며 데이터 분포를 학습하는 모델",
    inputShape: [4],
    palette: ["dense", "dropout", "batchNormalization", "activation"],
    branches: {
      generator: [
        { type: "dense", units: 16, activation: "relu" },
        { type: "batchNormalization" },
        { type: "dense", units: 16, activation: "relu" },
      ],
      discriminator: [
        { type: "dense", units: 16, activation: "leakyRelu" },
        { type: "dropout", rate: 0.2 },
        { type: "dense", units: 8, activation: "leakyRelu" },
      ],
    },
  },
};

const activationOptions = ACTIVATIONS.map((activation) => activation.value);

export const LAYER_CATALOG = {
  dense: {
    label: "Dense",
    category: "Core",
    description: "모든 입력과 뉴런을 연결하는 완전연결 레이어",
    fields: {
      units: { type: "number", label: "Units", min: 1, max: 256, step: 1, default: 8 },
      activation: { type: "select", label: "Activation", options: activationOptions, default: "relu" },
    },
  },
  dropout: {
    label: "Dropout",
    category: "Regularization",
    description: "학습 중 일부 활성화를 무작위로 비활성화",
    fields: {
      rate: { type: "number", label: "Rate", min: 0, max: 0.9, step: 0.05, default: 0.2 },
    },
  },
  batchNormalization: {
    label: "Batch Normalization",
    category: "Normalization",
    description: "미니배치 단위로 활성화 분포를 정규화",
    fields: {},
  },
  activation: {
    label: "Activation",
    category: "Activation",
    description: "별도 활성화 함수를 적용",
    fields: {
      activation: { type: "select", label: "Function", options: activationOptions, default: "relu" },
    },
  },
  conv2d: {
    label: "Conv2D",
    category: "CNN",
    description: "2D 이미지에 학습 가능한 합성곱 필터 적용",
    fields: {
      filters: { type: "number", label: "Filters", min: 1, max: 128, step: 1, default: 8 },
      kernelSize: { type: "number", label: "Kernel", min: 1, max: 7, step: 1, default: 3 },
      strides: { type: "number", label: "Stride", min: 1, max: 4, step: 1, default: 1 },
      padding: { type: "select", label: "Padding", options: ["same", "valid"], default: "same" },
      activation: { type: "select", label: "Activation", options: activationOptions, default: "relu" },
    },
  },
  conv2dTranspose: {
    label: "Conv2D Transpose",
    category: "CNN / GAN",
    description: "공간 해상도를 키우는 전치 합성곱",
    fields: {
      filters: { type: "number", label: "Filters", min: 1, max: 128, step: 1, default: 8 },
      kernelSize: { type: "number", label: "Kernel", min: 1, max: 7, step: 1, default: 3 },
      strides: { type: "number", label: "Stride", min: 1, max: 4, step: 1, default: 2 },
      padding: { type: "select", label: "Padding", options: ["same", "valid"], default: "same" },
      activation: { type: "select", label: "Activation", options: activationOptions, default: "relu" },
    },
  },
  maxPooling2d: {
    label: "Max Pooling 2D",
    category: "CNN",
    description: "영역 내 최대값으로 이미지 크기를 축소",
    fields: {
      poolSize: { type: "number", label: "Pool", min: 2, max: 4, step: 1, default: 2 },
      strides: { type: "number", label: "Stride", min: 1, max: 4, step: 1, default: 2 },
      padding: { type: "select", label: "Padding", options: ["same", "valid"], default: "valid" },
    },
  },
  averagePooling2d: {
    label: "Average Pooling 2D",
    category: "CNN",
    description: "영역 내 평균값으로 이미지 크기를 축소",
    fields: {
      poolSize: { type: "number", label: "Pool", min: 2, max: 4, step: 1, default: 2 },
      strides: { type: "number", label: "Stride", min: 1, max: 4, step: 1, default: 2 },
      padding: { type: "select", label: "Padding", options: ["same", "valid"], default: "valid" },
    },
  },
  globalAveragePooling2d: {
    label: "Global Average Pooling 2D",
    category: "CNN",
    description: "각 채널의 공간 평균을 하나의 값으로 축소",
    fields: {},
  },
  flatten: {
    label: "Flatten",
    category: "Shape",
    description: "다차원 feature map을 1차원 벡터로 변환",
    fields: {},
  },
  reshape: {
    label: "Reshape",
    category: "Shape",
    description: "원소 수를 유지하며 tensor shape 변경",
    fields: {
      targetShape: { type: "text", label: "Target shape", default: "4,4,1" },
    },
  },
  embedding: {
    label: "Embedding",
    category: "Sequence",
    description: "정수 token을 학습 가능한 벡터로 변환",
    fields: {
      inputDim: { type: "number", label: "Vocabulary", min: 2, max: 10000, step: 1, default: 128 },
      outputDim: { type: "number", label: "Embedding dim", min: 1, max: 256, step: 1, default: 16 },
      inputLength: { type: "number", label: "Sequence length", min: 1, max: 512, step: 1, default: 12 },
    },
  },
  simpleRnn: {
    label: "Simple RNN",
    category: "RNN",
    description: "이전 hidden state를 다음 시점으로 전달",
    fields: {
      units: { type: "number", label: "Units", min: 1, max: 128, step: 1, default: 12 },
      activation: { type: "select", label: "Activation", options: activationOptions, default: "tanh" },
      returnSequences: { type: "boolean", label: "Return sequences", default: false },
    },
  },
  lstm: {
    label: "LSTM",
    category: "RNN",
    description: "입력/출력/망각 gate로 장기 의존성 학습",
    fields: {
      units: { type: "number", label: "Units", min: 1, max: 128, step: 1, default: 12 },
      activation: { type: "select", label: "Activation", options: activationOptions, default: "tanh" },
      recurrentActivation: {
        type: "select",
        label: "Recurrent activation",
        options: ["sigmoid", "hardSigmoid", "tanh"],
        default: "sigmoid",
      },
      returnSequences: { type: "boolean", label: "Return sequences", default: false },
    },
  },
  gru: {
    label: "GRU",
    category: "RNN",
    description: "update/reset gate를 사용하는 recurrent layer",
    fields: {
      units: { type: "number", label: "Units", min: 1, max: 128, step: 1, default: 12 },
      activation: { type: "select", label: "Activation", options: activationOptions, default: "tanh" },
      recurrentActivation: {
        type: "select",
        label: "Recurrent activation",
        options: ["sigmoid", "hardSigmoid", "tanh"],
        default: "sigmoid",
      },
      returnSequences: { type: "boolean", label: "Return sequences", default: false },
    },
  },
};

export function createLayer(type, overrides = {}) {
  const definition = LAYER_CATALOG[type];
  if (!definition) throw new Error(`Unknown layer type: ${type}`);
  const layer = { id: crypto.randomUUID(), type };
  for (const [fieldName, field] of Object.entries(definition.fields)) {
    layer[fieldName] = overrides[fieldName] ?? field.default;
  }
  return { ...layer, ...overrides, id: overrides.id ?? layer.id };
}

export function cloneTemplate(family) {
  const definition = MODEL_FAMILIES[family];
  if (!definition) throw new Error(`Unknown model family: ${family}`);
  if (family === "gan") {
    return {
      generator: definition.branches.generator.map((layer) => createLayer(layer.type, layer)),
      discriminator: definition.branches.discriminator.map((layer) => createLayer(layer.type, layer)),
    };
  }
  return definition.template.map((layer) => createLayer(layer.type, layer));
}

function product(shape) {
  return shape.reduce((total, value) => total * value, 1);
}

function convOutput(size, kernel, stride, padding) {
  return padding === "same"
    ? Math.ceil(size / stride)
    : Math.floor((size - kernel) / stride) + 1;
}

function parseShape(value) {
  return String(value)
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isInteger(part) && part > 0);
}

export function inferLayerShape(inputShape, layer) {
  const shape = inputShape.slice();

  if (["dropout", "batchNormalization", "activation"].includes(layer.type)) return shape;

  if (layer.type === "dense") {
    if (shape.length < 1) throw new Error("Dense requires rank >= 1");
    return [...shape.slice(0, -1), Number(layer.units)];
  }

  if (["conv2d", "conv2dTranspose"].includes(layer.type)) {
    if (shape.length !== 3) throw new Error(`${layer.type} requires [height, width, channels]`);
    const [height, width] = shape;
    if (layer.type === "conv2dTranspose") {
      const outputHeight = layer.padding === "same"
        ? height * Number(layer.strides)
        : (height - 1) * Number(layer.strides) + Number(layer.kernelSize);
      const outputWidth = layer.padding === "same"
        ? width * Number(layer.strides)
        : (width - 1) * Number(layer.strides) + Number(layer.kernelSize);
      return [outputHeight, outputWidth, Number(layer.filters)];
    }
    const outputHeight = convOutput(
      height,
      Number(layer.kernelSize),
      Number(layer.strides),
      layer.padding,
    );
    const outputWidth = convOutput(
      width,
      Number(layer.kernelSize),
      Number(layer.strides),
      layer.padding,
    );
    if (outputHeight < 1 || outputWidth < 1) throw new Error("Kernel/pooling configuration collapses spatial shape");
    return [outputHeight, outputWidth, Number(layer.filters)];
  }

  if (["maxPooling2d", "averagePooling2d"].includes(layer.type)) {
    if (shape.length !== 3) throw new Error(`${layer.type} requires [height, width, channels]`);
    const outputHeight = convOutput(
      shape[0],
      Number(layer.poolSize),
      Number(layer.strides),
      layer.padding,
    );
    const outputWidth = convOutput(
      shape[1],
      Number(layer.poolSize),
      Number(layer.strides),
      layer.padding,
    );
    if (outputHeight < 1 || outputWidth < 1) throw new Error("Pooling configuration collapses spatial shape");
    return [outputHeight, outputWidth, shape[2]];
  }

  if (layer.type === "globalAveragePooling2d") {
    if (shape.length !== 3) throw new Error("GlobalAveragePooling2D requires rank 3");
    return [shape[2]];
  }

  if (layer.type === "flatten") return [product(shape)];

  if (layer.type === "reshape") {
    const targetShape = parseShape(layer.targetShape);
    if (!targetShape.length || product(targetShape) !== product(shape)) {
      throw new Error(`Reshape must preserve ${product(shape)} values`);
    }
    return targetShape;
  }

  if (layer.type === "embedding") {
    if (shape.length !== 1) throw new Error("Embedding requires a token sequence shape");
    return [Number(layer.inputLength), Number(layer.outputDim)];
  }

  if (["simpleRnn", "lstm", "gru"].includes(layer.type)) {
    if (shape.length !== 2) throw new Error(`${layer.type} requires [timesteps, features]`);
    return layer.returnSequences
      ? [shape[0], Number(layer.units)]
      : [Number(layer.units)];
  }

  throw new Error(`Shape inference is not implemented for ${layer.type}`);
}

export function validateArchitecture(inputShape, layers) {
  const rows = [];
  let shape = inputShape.slice();
  let error = null;

  for (const layer of layers) {
    const input = shape.slice();
    try {
      shape = inferLayerShape(shape, layer);
      rows.push({ layerId: layer.id, input, output: shape.slice(), valid: true });
    } catch (layerError) {
      error = layerError.message;
      rows.push({ layerId: layer.id, input, output: null, valid: false, error });
      break;
    }
  }

  return {
    valid: !error,
    error,
    outputShape: error ? null : shape,
    rows,
  };
}

export function formatShape(shape) {
  return shape ? `[${shape.join(", ")}]` : "invalid";
}
