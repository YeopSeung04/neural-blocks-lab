import {
  MODEL_FAMILIES,
  validateArchitecture,
} from "./layer-catalog.mjs";

function parseShape(value) {
  return String(value)
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isInteger(part) && part > 0);
}

function withInputShape(options, inputShape, isFirst) {
  return isFirst ? { ...options, inputShape } : options;
}

function coreActivation(activation) {
  return activation === "leakyRelu" ? "linear" : activation;
}

function addPostActivation(tf, model, activation) {
  if (activation === "leakyRelu") {
    model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
  }
}

function addConfiguredLayer(tf, model, layer, inputShape, isFirst) {
  const activation = layer.activation ?? "linear";

  if (layer.type === "dense") {
    model.add(tf.layers.dense(withInputShape({
      units: Number(layer.units),
      activation: coreActivation(activation),
    }, inputShape, isFirst)));
    addPostActivation(tf, model, activation);
    return;
  }

  if (layer.type === "dropout") {
    model.add(tf.layers.dropout(withInputShape({ rate: Number(layer.rate) }, inputShape, isFirst)));
    return;
  }

  if (layer.type === "batchNormalization") {
    model.add(tf.layers.batchNormalization(withInputShape({}, inputShape, isFirst)));
    return;
  }

  if (layer.type === "activation") {
    if (layer.activation === "leakyRelu") {
      model.add(tf.layers.leakyReLU(withInputShape({ alpha: 0.2 }, inputShape, isFirst)));
    } else {
      model.add(tf.layers.activation(withInputShape({
        activation: layer.activation,
      }, inputShape, isFirst)));
    }
    return;
  }

  if (layer.type === "conv2d") {
    model.add(tf.layers.conv2d(withInputShape({
      filters: Number(layer.filters),
      kernelSize: Number(layer.kernelSize),
      strides: Number(layer.strides),
      padding: layer.padding,
      activation: coreActivation(activation),
    }, inputShape, isFirst)));
    addPostActivation(tf, model, activation);
    return;
  }

  if (layer.type === "conv2dTranspose") {
    model.add(tf.layers.conv2dTranspose(withInputShape({
      filters: Number(layer.filters),
      kernelSize: Number(layer.kernelSize),
      strides: Number(layer.strides),
      padding: layer.padding,
      activation: coreActivation(activation),
    }, inputShape, isFirst)));
    addPostActivation(tf, model, activation);
    return;
  }

  if (layer.type === "maxPooling2d") {
    model.add(tf.layers.maxPooling2d(withInputShape({
      poolSize: [Number(layer.poolSize), Number(layer.poolSize)],
      strides: [Number(layer.strides), Number(layer.strides)],
      padding: layer.padding,
    }, inputShape, isFirst)));
    return;
  }

  if (layer.type === "averagePooling2d") {
    model.add(tf.layers.averagePooling2d(withInputShape({
      poolSize: [Number(layer.poolSize), Number(layer.poolSize)],
      strides: [Number(layer.strides), Number(layer.strides)],
      padding: layer.padding,
    }, inputShape, isFirst)));
    return;
  }

  if (layer.type === "globalAveragePooling2d") {
    model.add(tf.layers.globalAveragePooling2d(withInputShape({}, inputShape, isFirst)));
    return;
  }

  if (layer.type === "flatten") {
    model.add(tf.layers.flatten(withInputShape({}, inputShape, isFirst)));
    return;
  }

  if (layer.type === "reshape") {
    model.add(tf.layers.reshape(withInputShape({
      targetShape: parseShape(layer.targetShape),
    }, inputShape, isFirst)));
    return;
  }

  if (layer.type === "embedding") {
    model.add(tf.layers.embedding(withInputShape({
      inputDim: Number(layer.inputDim),
      outputDim: Number(layer.outputDim),
      inputLength: Number(layer.inputLength),
    }, inputShape, isFirst)));
    return;
  }

  if (layer.type === "simpleRnn") {
    model.add(tf.layers.simpleRNN(withInputShape({
      units: Number(layer.units),
      activation: coreActivation(activation),
      returnSequences: Boolean(layer.returnSequences),
    }, inputShape, isFirst)));
    addPostActivation(tf, model, activation);
    return;
  }

  if (layer.type === "lstm") {
    model.add(tf.layers.lstm(withInputShape({
      units: Number(layer.units),
      activation: coreActivation(activation),
      recurrentActivation: layer.recurrentActivation,
      returnSequences: Boolean(layer.returnSequences),
    }, inputShape, isFirst)));
    addPostActivation(tf, model, activation);
    return;
  }

  if (layer.type === "gru") {
    model.add(tf.layers.gru(withInputShape({
      units: Number(layer.units),
      activation: coreActivation(activation),
      recurrentActivation: layer.recurrentActivation,
      returnSequences: Boolean(layer.returnSequences),
    }, inputShape, isFirst)));
    addPostActivation(tf, model, activation);
    return;
  }

  throw new Error(`TensorFlow.js builder does not support ${layer.type}`);
}

export function buildClassifier(tf, family, layers, inputShapeOverride = null) {
  const familyDefinition = MODEL_FAMILIES[family];
  if (!familyDefinition || family === "gan") {
    throw new Error(`Classifier family is not supported: ${family}`);
  }
  const inputShape = inputShapeOverride ?? familyDefinition.inputShape;
  const validation = validateArchitecture(inputShape, layers);
  if (!validation.valid) throw new Error(validation.error);
  if (validation.outputShape.length !== 1) {
    throw new Error(
      `Binary classification head requires a vector, received [${validation.outputShape.join(", ")}]`,
    );
  }

  const model = tf.sequential();
  layers.forEach((layer, index) => {
    addConfiguredLayer(tf, model, layer, inputShape, index === 0);
  });
  model.add(tf.layers.dense({
    units: 1,
    activation: "sigmoid",
    ...(layers.length === 0 ? { inputShape } : {}),
  }));
  return model;
}

function buildGanBranch(tf, inputShape, layers, outputUnits, outputActivation) {
  const validation = validateArchitecture(inputShape, layers);
  if (!validation.valid) throw new Error(validation.error);
  if (validation.outputShape.length !== 1) {
    throw new Error(`GAN branch requires a vector, received [${validation.outputShape.join(", ")}]`);
  }
  const model = tf.sequential();
  layers.forEach((layer, index) => {
    addConfiguredLayer(tf, model, layer, inputShape, index === 0);
  });
  model.add(tf.layers.dense({
    units: outputUnits,
    activation: outputActivation,
    ...(layers.length === 0 ? { inputShape } : {}),
  }));
  return model;
}

export function buildGan(tf, branches) {
  return {
    generator: buildGanBranch(tf, [4], branches.generator, 2, "tanh"),
    discriminator: buildGanBranch(tf, [2], branches.discriminator, 1, "sigmoid"),
  };
}

export function countParameters(model) {
  return model.countParams();
}
