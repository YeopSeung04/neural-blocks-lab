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
