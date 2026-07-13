const EPSILON = 1e-8;

export function createRng(seed = 123456789) {
  let state = seed >>> 0;
  return function random() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  const u = Math.max(random(), EPSILON);
  const v = Math.max(random(), EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function shuffle(values, random) {
  const copy = values.slice();
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [copy[index], copy[target]] = [copy[target], copy[index]];
  }
  return copy;
}

export function makeDataset(type, count = 240, noise = 0.12, seed = 42) {
  const random = createRng(seed);
  const points = [];

  for (let index = 0; index < count; index += 1) {
    let x;
    let y;
    let label;

    if (type === "circles") {
      label = index % 2;
      const baseRadius = label ? 0.72 : 0.32;
      const radius = baseRadius + gaussian(random) * noise * 0.35;
      const angle = random() * Math.PI * 2;
      x = Math.cos(angle) * radius + gaussian(random) * noise * 0.18;
      y = Math.sin(angle) * radius + gaussian(random) * noise * 0.18;
    } else if (type === "linear") {
      x = random() * 2 - 1;
      y = random() * 2 - 1;
      label = x * 0.8 + y * 0.65 + gaussian(random) * noise > 0 ? 1 : 0;
    } else {
      x = random() * 2 - 1;
      y = random() * 2 - 1;
      const cleanLabel = (x > 0) !== (y > 0) ? 1 : 0;
      label = random() < noise * 0.22 ? 1 - cleanLabel : cleanLabel;
    }

    points.push({ input: [x, y], target: label });
  }

  return points;
}

export function splitDataset(points, validationRatio = 0.25, seed = 777) {
  const random = createRng(seed);
  const shuffled = shuffle(points, random);
  const validationCount = Math.max(1, Math.floor(points.length * validationRatio));
  return {
    validation: shuffled.slice(0, validationCount),
    train: shuffled.slice(validationCount),
  };
}

const activations = {
  relu: {
    forward: (value) => Math.max(0, value),
    derivative: (value) => (value > 0 ? 1 : 0),
  },
  tanh: {
    forward: (value) => Math.tanh(value),
    derivative: (value) => {
      const output = Math.tanh(value);
      return 1 - output * output;
    },
  },
  sigmoid: {
    forward: (value) => {
      if (value >= 40) return 1;
      if (value <= -40) return 0;
      return 1 / (1 + Math.exp(-value));
    },
    derivative: (value) => {
      const output = activations.sigmoid.forward(value);
      return output * (1 - output);
    },
  },
  linear: {
    forward: (value) => value,
    derivative: () => 1,
  },
};

function makeMatrix(rows, columns, fill = 0) {
  return Array.from({ length: rows }, () => Array(columns).fill(fill));
}

function cloneMatrix(matrix) {
  return matrix.map((row) => row.slice());
}

function zerosLikeLayer(layer) {
  return {
    weights: makeMatrix(layer.weights.length, layer.weights[0].length),
    biases: Array(layer.biases.length).fill(0),
  };
}

export class NeuralNetwork {
  constructor(hiddenLayers = [], seed = 1) {
    this.hiddenLayers = hiddenLayers.map((layer) => ({ ...layer }));
    this.seed = seed;
    this.random = createRng(seed);
    this.layers = [];
    this.optimizerState = null;
    this.optimizerStep = 0;
    this.build();
  }

  build() {
    const definitions = [
      ...this.hiddenLayers.map((layer) => ({
        units: Math.max(1, Math.round(layer.units)),
        activation: layer.activation,
      })),
      { units: 1, activation: "sigmoid" },
    ];

    this.layers = [];
    let inputSize = 2;

    for (const definition of definitions) {
      const activation = activations[definition.activation] ? definition.activation : "relu";
      const scale = activation === "relu"
        ? Math.sqrt(2 / inputSize)
        : Math.sqrt(1 / inputSize);
      const weights = Array.from({ length: definition.units }, () =>
        Array.from({ length: inputSize }, () => gaussian(this.random) * scale),
      );
      this.layers.push({
        inputSize,
        units: definition.units,
        activation,
        weights,
        biases: Array(definition.units).fill(0),
      });
      inputSize = definition.units;
    }

    this.resetOptimizerState();
  }

  resetOptimizerState() {
    this.optimizerStep = 0;
    this.optimizerState = {
      velocity: this.layers.map(zerosLikeLayer),
      mean: this.layers.map(zerosLikeLayer),
      square: this.layers.map(zerosLikeLayer),
    };
  }

  cloneParameters() {
    return this.layers.map((layer) => ({
      weights: cloneMatrix(layer.weights),
      biases: layer.biases.slice(),
    }));
  }

  forward(input, capture = false) {
    let current = input.slice();
    const cache = [];

    for (const layer of this.layers) {
      const z = layer.weights.map((weights, unitIndex) => {
        let total = layer.biases[unitIndex];
        for (let inputIndex = 0; inputIndex < weights.length; inputIndex += 1) {
          total += weights[inputIndex] * current[inputIndex];
        }
        return total;
      });
      const output = z.map(activations[layer.activation].forward);
      if (capture) {
        cache.push({
          input: current.slice(),
          z,
          output: output.slice(),
        });
      }
      current = output;
    }

    return capture ? { output: current[0], cache } : current[0];
  }

  predict(input) {
    return this.forward(input, false);
  }

  loss(points) {
    let total = 0;
    for (const point of points) {
      const prediction = Math.min(Math.max(this.predict(point.input), EPSILON), 1 - EPSILON);
      total += -(point.target * Math.log(prediction) + (1 - point.target) * Math.log(1 - prediction));
    }
    return total / points.length;
  }

  evaluate(points) {
    let totalLoss = 0;
    let correct = 0;
    for (const point of points) {
      const prediction = Math.min(Math.max(this.predict(point.input), EPSILON), 1 - EPSILON);
      totalLoss += -(point.target * Math.log(prediction) + (1 - point.target) * Math.log(1 - prediction));
      if ((prediction >= 0.5 ? 1 : 0) === point.target) correct += 1;
    }
    return {
      loss: totalLoss / points.length,
      accuracy: correct / points.length,
    };
  }

  gradients(batch) {
    const gradients = this.layers.map(zerosLikeLayer);

    for (const point of batch) {
      const { output, cache } = this.forward(point.input, true);
      let delta = [output - point.target];

      for (let layerIndex = this.layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
        const layer = this.layers[layerIndex];
        const layerCache = cache[layerIndex];
        const layerGradient = gradients[layerIndex];

        for (let unitIndex = 0; unitIndex < layer.units; unitIndex += 1) {
          layerGradient.biases[unitIndex] += delta[unitIndex];
          for (let inputIndex = 0; inputIndex < layer.inputSize; inputIndex += 1) {
            layerGradient.weights[unitIndex][inputIndex] +=
              delta[unitIndex] * layerCache.input[inputIndex];
          }
        }

        if (layerIndex > 0) {
          const previousLayer = this.layers[layerIndex - 1];
          const previousCache = cache[layerIndex - 1];
          const nextDelta = Array(previousLayer.units).fill(0);

          for (let previousUnit = 0; previousUnit < previousLayer.units; previousUnit += 1) {
            let propagated = 0;
            for (let unitIndex = 0; unitIndex < layer.units; unitIndex += 1) {
              propagated += layer.weights[unitIndex][previousUnit] * delta[unitIndex];
            }
            nextDelta[previousUnit] =
              propagated * activations[previousLayer.activation].derivative(previousCache.z[previousUnit]);
          }
          delta = nextDelta;
        }
      }
    }

    const scale = 1 / batch.length;
    for (const gradient of gradients) {
      for (let unitIndex = 0; unitIndex < gradient.weights.length; unitIndex += 1) {
        gradient.biases[unitIndex] *= scale;
        for (let inputIndex = 0; inputIndex < gradient.weights[unitIndex].length; inputIndex += 1) {
          gradient.weights[unitIndex][inputIndex] *= scale;
        }
      }
    }
    return gradients;
  }

  applyGradients(gradients, optimizer = "adam", learningRate = 0.03) {
    this.optimizerStep += 1;

    for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex += 1) {
      const layer = this.layers[layerIndex];
      const gradient = gradients[layerIndex];

      for (let unitIndex = 0; unitIndex < layer.units; unitIndex += 1) {
        for (let inputIndex = 0; inputIndex < layer.inputSize; inputIndex += 1) {
          const value = gradient.weights[unitIndex][inputIndex];
          layer.weights[unitIndex][inputIndex] -= this.optimizerDelta(
            optimizer,
            learningRate,
            value,
            this.optimizerState.velocity[layerIndex].weights[unitIndex],
            this.optimizerState.mean[layerIndex].weights[unitIndex],
            this.optimizerState.square[layerIndex].weights[unitIndex],
            inputIndex,
          );
        }

        const biasGradient = gradient.biases[unitIndex];
        layer.biases[unitIndex] -= this.optimizerDelta(
          optimizer,
          learningRate,
          biasGradient,
          this.optimizerState.velocity[layerIndex].biases,
          this.optimizerState.mean[layerIndex].biases,
          this.optimizerState.square[layerIndex].biases,
          unitIndex,
        );
      }
    }
  }

  optimizerDelta(optimizer, learningRate, gradient, velocity, mean, square, index) {
    if (optimizer === "sgd") return learningRate * gradient;

    if (optimizer === "momentum") {
      velocity[index] = 0.9 * velocity[index] + gradient;
      return learningRate * velocity[index];
    }

    if (optimizer === "rmsprop") {
      square[index] = 0.99 * square[index] + 0.01 * gradient * gradient;
      return learningRate * gradient / (Math.sqrt(square[index]) + EPSILON);
    }

    mean[index] = 0.9 * mean[index] + 0.1 * gradient;
    square[index] = 0.999 * square[index] + 0.001 * gradient * gradient;
    const meanHat = mean[index] / (1 - 0.9 ** this.optimizerStep);
    const squareHat = square[index] / (1 - 0.999 ** this.optimizerStep);
    return learningRate * meanHat / (Math.sqrt(squareHat) + EPSILON);
  }

  trainBatch(batch, optimizer, learningRate) {
    const gradients = this.gradients(batch);
    this.applyGradients(gradients, optimizer, learningRate);
  }

  architecture() {
    return [
      { type: "input", units: 2, activation: "linear" },
      ...this.layers.map((layer, index) => ({
        type: index === this.layers.length - 1 ? "output" : "hidden",
        units: layer.units,
        activation: layer.activation,
      })),
    ];
  }

  activationSnapshot(input) {
    const result = this.forward(input, true);
    return {
      input: input.slice(),
      layers: result.cache.map((entry) => entry.output.slice()),
      output: result.output,
    };
  }
}

export class TrainingSession {
  constructor({
    dataset = "xor",
    count = 240,
    noise = 0.12,
    hiddenLayers = [{ units: 8, activation: "tanh" }],
    optimizer = "adam",
    learningRate = 0.03,
    batchSize = 24,
    seed = 11,
  } = {}) {
    this.config = {
      dataset,
      count,
      noise,
      hiddenLayers,
      optimizer,
      learningRate,
      batchSize,
      seed,
    };
    this.reset();
  }

  reset(nextConfig = {}) {
    this.config = {
      ...this.config,
      ...nextConfig,
      hiddenLayers: (nextConfig.hiddenLayers ?? this.config.hiddenLayers).map((layer) => ({ ...layer })),
    };
    this.random = createRng(this.config.seed);
    this.points = makeDataset(
      this.config.dataset,
      this.config.count,
      this.config.noise,
      this.config.seed,
    );
    const split = splitDataset(this.points, 0.25, this.config.seed + 1);
    this.trainPoints = split.train;
    this.validationPoints = split.validation;
    this.network = new NeuralNetwork(this.config.hiddenLayers, this.config.seed + 2);
    this.order = shuffle(
      Array.from({ length: this.trainPoints.length }, (_, index) => index),
      this.random,
    );
    this.cursor = 0;
    this.epoch = 0;
    this.steps = 0;
    this.history = [];
    this.recordMetrics();
  }

  updateTrainingConfig(nextConfig) {
    this.config = { ...this.config, ...nextConfig };
  }

  nextBatch() {
    if (this.cursor >= this.order.length) {
      this.epoch += 1;
      this.order = shuffle(
        Array.from({ length: this.trainPoints.length }, (_, index) => index),
        this.random,
      );
      this.cursor = 0;
      this.recordMetrics();
    }
    const batchSize = Math.min(this.config.batchSize, this.trainPoints.length);
    const indices = this.order.slice(this.cursor, this.cursor + batchSize);
    this.cursor += indices.length;
    return indices.map((index) => this.trainPoints[index]);
  }

  step(updateCount = 1) {
    for (let index = 0; index < updateCount; index += 1) {
      const batch = this.nextBatch();
      this.network.trainBatch(batch, this.config.optimizer, this.config.learningRate);
      this.steps += 1;
    }
  }

  recordMetrics() {
    const train = this.network.evaluate(this.trainPoints);
    const validation = this.network.evaluate(this.validationPoints);
    this.history.push({
      epoch: this.epoch,
      trainLoss: train.loss,
      validationLoss: validation.loss,
      trainAccuracy: train.accuracy,
      validationAccuracy: validation.accuracy,
    });
    if (this.history.length > 1200) this.history.shift();
  }

  metrics() {
    return {
      train: this.network.evaluate(this.trainPoints),
      validation: this.network.evaluate(this.validationPoints),
      epoch: this.epoch,
      steps: this.steps,
    };
  }
}
