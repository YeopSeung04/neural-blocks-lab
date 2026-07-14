import assert from "node:assert/strict";
import { TrainingSession } from "./ml-core.mjs";

function trainAndCheck(dataset, hiddenLayers, steps, expectedAccuracy) {
  const session = new TrainingSession({
    dataset,
    count: 240,
    noise: 0.08,
    hiddenLayers,
    optimizer: "adam",
    learningRate: 0.025,
    batchSize: 24,
    seed: 19,
  });

  const before = session.metrics();
  session.step(steps);
  session.recordMetrics();
  const after = session.metrics();

  assert.ok(
    after.train.loss < before.train.loss,
    `${dataset}: train loss did not decrease`,
  );
  assert.ok(
    after.validation.accuracy >= expectedAccuracy,
    `${dataset}: validation accuracy ${after.validation.accuracy} was below ${expectedAccuracy}`,
  );

  return {
    dataset,
    beforeLoss: before.train.loss,
    afterLoss: after.train.loss,
    validationAccuracy: after.validation.accuracy,
  };
}

const results = [
  trainAndCheck("linear", [], 600, 0.85),
  trainAndCheck("xor", [{ units: 8, activation: "tanh" }], 1600, 0.82),
  trainAndCheck(
    "circles",
    [
      { units: 10, activation: "tanh" },
      { units: 6, activation: "tanh" },
    ],
    2200,
    0.82,
  ),
];

for (const result of results) {
  console.log(
    `${result.dataset}: loss ${result.beforeLoss.toFixed(4)} -> ` +
      `${result.afterLoss.toFixed(4)}, val accuracy ${(result.validationAccuracy * 100).toFixed(1)}%`,
  );
}

const uploadedPoints = Array.from({ length: 40 }, (_, index) => {
  const x = index / 20 - 1;
  const y = Math.sin(index) * 0.1;
  return { input: [x, y], target: x > 0 ? 1 : 0 };
});
const uploadedSession = new TrainingSession({
  points: uploadedPoints,
  hiddenLayers: [],
  validationRatio: 0.2,
  learningRate: 0.04,
  batchSize: 8,
});
assert.equal(uploadedSession.points.length, uploadedPoints.length);
assert.equal(uploadedSession.validationPoints.length, 8);
uploadedSession.step(200);
assert.ok(Number.isFinite(uploadedSession.metrics().train.loss));
console.log("uploaded MLP points: custom dataset split and training passed");

const multiFeatureSplits = {
  train: Array.from({ length: 24 }, (_, index) => ({
    input: [index / 24, index % 2, Math.sin(index)],
    target: index >= 12 ? 1 : 0,
  })),
  validation: Array.from({ length: 6 }, (_, index) => ({
    input: [index / 6, index % 2, Math.cos(index)],
    target: index >= 3 ? 1 : 0,
  })),
  test: Array.from({ length: 6 }, (_, index) => ({
    input: [index / 6, (index + 1) % 2, Math.sin(index / 2)],
    target: index >= 3 ? 1 : 0,
  })),
};
const multiFeatureSession = new TrainingSession({
  pointSplits: multiFeatureSplits,
  inputSize: 3,
  hiddenLayers: [{ units: 4, activation: "tanh" }],
  batchSize: 6,
});
assert.equal(multiFeatureSession.network.architecture()[0].units, 3);
assert.equal(multiFeatureSession.testPoints.length, 6);
assert.ok(Number.isFinite(multiFeatureSession.metrics().test.loss));
console.log("uploaded MLP: explicit train/validation/test splits and 3 features passed");
