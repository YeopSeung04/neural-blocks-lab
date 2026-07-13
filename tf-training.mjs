import { buildClassifier, buildGan } from "./tf-models.mjs";

function seededRandom(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(random) {
  const u = Math.max(random(), 1e-8);
  const v = Math.max(random(), 1e-8);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeOptimizer(tf, name, learningRate) {
  if (name === "sgd") return tf.train.sgd(learningRate);
  if (name === "momentum") return tf.train.momentum(learningRate, 0.9);
  if (name === "rmsprop") return tf.train.rmsprop(learningRate);
  return tf.train.adam(learningRate);
}

function makeCnnArrays(count, seed) {
  const random = seededRandom(seed);
  const xs = new Float32Array(count * 8 * 8);
  const ys = new Float32Array(count);

  for (let sample = 0; sample < count; sample += 1) {
    const label = sample % 2;
    ys[sample] = label;
    const line = 2 + Math.floor(random() * 4);
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const onPattern = label === 0 ? Math.abs(x - line) <= 1 : Math.abs(y - line) <= 1;
        const noise = gaussian(random) * 0.12;
        xs[sample * 64 + y * 8 + x] = Math.max(0, Math.min(1, (onPattern ? 0.9 : 0.08) + noise));
      }
    }
  }
  return { xs, ys };
}

function makeRnnArrays(count, seed) {
  const random = seededRandom(seed);
  const xs = new Float32Array(count * 12);
  const ys = new Float32Array(count);

  for (let sample = 0; sample < count; sample += 1) {
    const label = sample % 2;
    ys[sample] = label;
    const start = random() * 0.5 - 0.25;
    const slope = label === 1 ? 0.07 : -0.07;
    for (let step = 0; step < 12; step += 1) {
      xs[sample * 12 + step] = start + slope * step + gaussian(random) * 0.08;
    }
  }
  return { xs, ys };
}

function splitTensorDataset(tf, family, count, seed) {
  const validationCount = Math.max(8, Math.floor(count * 0.2));
  const trainCount = count - validationCount;
  const data = family === "cnn"
    ? makeCnnArrays(count, seed)
    : makeRnnArrays(count, seed);
  const fullXs = family === "cnn"
    ? tf.tensor4d(data.xs, [count, 8, 8, 1])
    : tf.tensor3d(data.xs, [count, 12, 1]);
  const fullYs = tf.tensor2d(data.ys, [count, 1]);
  const trainXs = fullXs.slice([0, ...Array(fullXs.rank - 1).fill(0)], [trainCount, ...fullXs.shape.slice(1)]);
  const validationXs = fullXs.slice(
    [trainCount, ...Array(fullXs.rank - 1).fill(0)],
    [validationCount, ...fullXs.shape.slice(1)],
  );
  const trainYs = fullYs.slice([0, 0], [trainCount, 1]);
  const validationYs = fullYs.slice([trainCount, 0], [validationCount, 1]);
  fullXs.dispose();
  fullYs.dispose();
  return { trainXs, trainYs, validationXs, validationYs };
}

export class TfClassifierSession {
  constructor(tf, family, layers, config = {}) {
    const { model, ...trainingConfig } = config;
    this.tf = tf;
    this.family = family;
    this.layers = layers;
    this.config = {
      count: 160,
      optimizer: "adam",
      learningRate: 0.01,
      batchSize: 24,
      seed: 91,
      ...trainingConfig,
    };
    this.model = model ?? buildClassifier(tf, family, layers);
    this.data = splitTensorDataset(tf, family, this.config.count, this.config.seed);
    this.history = [];
    this.epoch = 0;
    this.busy = false;
    this.disposed = false;
    this.disposeRequested = false;
    this.compile();
  }

  compile() {
    this.optimizer?.dispose?.();
    this.optimizer = makeOptimizer(this.tf, this.config.optimizer, this.config.learningRate);
    this.model.compile({
      optimizer: this.optimizer,
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
  }

  updateConfig(nextConfig) {
    const shouldCompile =
      (nextConfig.optimizer !== undefined && nextConfig.optimizer !== this.config.optimizer) ||
      (
        nextConfig.learningRate !== undefined &&
        nextConfig.learningRate !== this.config.learningRate
      );
    this.config = { ...this.config, ...nextConfig };
    if (shouldCompile) this.compile();
  }

  async step(epochCount = 1) {
    if (this.busy || this.disposed || this.disposeRequested) return;
    this.busy = true;
    try {
      const result = await this.model.fit(this.data.trainXs, this.data.trainYs, {
        epochs: epochCount,
        batchSize: Math.min(this.config.batchSize, this.data.trainXs.shape[0]),
        validationData: [this.data.validationXs, this.data.validationYs],
        shuffle: true,
        verbose: 0,
      });
      for (let index = 0; index < result.epoch.length; index += 1) {
        this.epoch += 1;
        this.history.push({
          epoch: this.epoch,
          trainLoss: result.history.loss[index],
          validationLoss: result.history.val_loss[index],
          trainAccuracy: result.history.acc?.[index] ?? result.history.accuracy?.[index] ?? 0,
          validationAccuracy:
            result.history.val_acc?.[index] ?? result.history.val_accuracy?.[index] ?? 0,
        });
      }
    } finally {
      this.busy = false;
      if (this.disposeRequested) this.disposeResources();
    }
  }

  metrics() {
    return this.history[this.history.length - 1] ?? {
      epoch: 0,
      trainLoss: NaN,
      validationLoss: NaN,
      trainAccuracy: NaN,
      validationAccuracy: NaN,
    };
  }

  disposeResources() {
    if (this.disposed) return;
    this.disposed = true;
    this.model.dispose();
    this.optimizer?.dispose?.();
    for (const tensor of Object.values(this.data)) tensor.dispose();
  }

  dispose() {
    if (this.busy) {
      this.disposeRequested = true;
      return;
    }
    this.disposeResources();
  }
}

function sampleRealPoints(tf, count, seedOffset = 0) {
  const random = seededRandom(313 + seedOffset);
  const values = new Float32Array(count * 2);
  for (let index = 0; index < count; index += 1) {
    const angle = random() * Math.PI * 2;
    const radius = 0.65 + gaussian(random) * 0.05;
    values[index * 2] = Math.cos(angle) * radius;
    values[index * 2 + 1] = Math.sin(angle) * radius;
  }
  return tf.tensor2d(values, [count, 2]);
}

export class GanTrainingSession {
  constructor(tf, branches, config = {}) {
    const { models, ...trainingConfig } = config;
    this.tf = tf;
    this.config = {
      optimizer: "adam",
      learningRate: 0.002,
      batchSize: 32,
      ...trainingConfig,
    };
    const compiledModels = models ?? buildGan(tf, branches);
    this.generator = compiledModels.generator;
    this.discriminator = compiledModels.discriminator;
    this.history = [];
    this.epoch = 0;
    this.busy = false;
    this.compile();
  }

  compile() {
    this.generatorOptimizer?.dispose?.();
    this.discriminatorOptimizer?.dispose?.();
    this.generatorOptimizer = makeOptimizer(
      this.tf,
      this.config.optimizer,
      this.config.learningRate,
    );
    this.discriminatorOptimizer = makeOptimizer(
      this.tf,
      this.config.optimizer,
      this.config.learningRate,
    );
  }

  updateConfig(nextConfig) {
    const shouldCompile =
      (nextConfig.optimizer !== undefined && nextConfig.optimizer !== this.config.optimizer) ||
      (
        nextConfig.learningRate !== undefined &&
        nextConfig.learningRate !== this.config.learningRate
      );
    this.config = { ...this.config, ...nextConfig };
    if (shouldCompile) this.compile();
  }

  async step(stepCount = 1) {
    if (this.busy) return;
    this.busy = true;
    try {
      for (let step = 0; step < stepCount; step += 1) {
        const batchSize = this.config.batchSize;
        const discriminatorVariables = this.discriminator.trainableWeights.map((weight) => weight.val);
        const generatorVariables = this.generator.trainableWeights.map((weight) => weight.val);

        const discriminatorLoss = this.discriminatorOptimizer.minimize(() =>
          this.tf.tidy(() => {
            const real = sampleRealPoints(this.tf, batchSize, this.epoch + step);
            const noise = this.tf.randomNormal([batchSize, 4]);
            const fake = this.generator.apply(noise);
            const realPrediction = this.discriminator.apply(real);
            const fakePrediction = this.discriminator.apply(fake);
            const realLoss = this.tf.losses.logLoss(this.tf.onesLike(realPrediction), realPrediction);
            const fakeLoss = this.tf.losses.logLoss(this.tf.zerosLike(fakePrediction), fakePrediction);
            return realLoss.add(fakeLoss);
          }), true, discriminatorVariables);

        const generatorLoss = this.generatorOptimizer.minimize(() =>
          this.tf.tidy(() => {
            const noise = this.tf.randomNormal([batchSize, 4]);
            const fake = this.generator.apply(noise);
            const prediction = this.discriminator.apply(fake);
            return this.tf.losses.logLoss(this.tf.onesLike(prediction), prediction);
          }), true, generatorVariables);

        const discriminatorValue = discriminatorLoss.dataSync()[0];
        const generatorValue = generatorLoss.dataSync()[0];
        discriminatorLoss.dispose();
        generatorLoss.dispose();
        this.epoch += 1;
        this.history.push({
          epoch: this.epoch,
          trainLoss: generatorValue,
          validationLoss: discriminatorValue,
          trainAccuracy: NaN,
          validationAccuracy: NaN,
        });
      }
    } finally {
      this.busy = false;
    }
  }

  metrics() {
    return this.history[this.history.length - 1] ?? {
      epoch: 0,
      trainLoss: NaN,
      validationLoss: NaN,
      trainAccuracy: NaN,
      validationAccuracy: NaN,
    };
  }

  generatedPoints(count = 120) {
    return this.tf.tidy(() => {
      const points = this.generator.predict(this.tf.randomNormal([count, 4]));
      return points.arraySync();
    });
  }

  dispose() {
    this.generator.dispose();
    this.discriminator.dispose();
    this.generatorOptimizer?.dispose?.();
    this.discriminatorOptimizer?.dispose?.();
  }
}
