import assert from "node:assert/strict";
import * as tf from "@tensorflow/tfjs";
import { cloneTemplate } from "./layer-catalog.mjs";
import {
  GanTrainingSession,
  TfClassifierSession,
} from "./tf-training.mjs";

for (const family of ["cnn", "rnn"]) {
  const session = new TfClassifierSession(tf, family, cloneTemplate(family), {
    count: 120,
    learningRate: 0.01,
    batchSize: 24,
  });
  await session.step(1);
  const first = session.metrics();
  await session.step(7);
  const last = session.metrics();
  assert.ok(Number.isFinite(first.trainLoss));
  assert.ok(last.trainLoss < first.trainLoss, `${family} loss should decrease`);
  console.log(
    `${family}: loss ${first.trainLoss.toFixed(4)} -> ${last.trainLoss.toFixed(4)}, ` +
      `val accuracy ${(last.validationAccuracy * 100).toFixed(1)}%`,
  );
  session.dispose();
}

const customSequenceCount = 32;
const customSequenceLength = 8;
const customSequenceXs = new Float32Array(customSequenceCount * customSequenceLength);
const customSequenceYs = new Float32Array(customSequenceCount);
for (let sample = 0; sample < customSequenceCount; sample += 1) {
  const label = sample % 2;
  customSequenceYs[sample] = label;
  for (let step = 0; step < customSequenceLength; step += 1) {
    customSequenceXs[sample * customSequenceLength + step] =
      label ? step / customSequenceLength : 1 - step / customSequenceLength;
  }
}
const customRnn = new TfClassifierSession(tf, "rnn", cloneTemplate("rnn"), {
  datasetData: {
    count: customSequenceCount,
    inputShape: [customSequenceLength, 1],
    xs: customSequenceXs,
    ys: customSequenceYs,
  },
  inputShape: [customSequenceLength, 1],
  validationRatio: 0.25,
  batchSize: 8,
});
await customRnn.step(1);
assert.ok(Number.isFinite(customRnn.metrics().trainLoss));
assert.deepEqual(customRnn.model.inputs[0].shape, [null, customSequenceLength, 1]);
customRnn.dispose();
console.log("rnn: uploaded tensor dataset and dynamic input shape passed");

const deferredDisposeSession = new TfClassifierSession(
  tf,
  "cnn",
  cloneTemplate("cnn"),
  { count: 80, batchSize: 16 },
);
const pendingStep = deferredDisposeSession.step(1);
deferredDisposeSession.dispose();
await pendingStep;
assert.equal(deferredDisposeSession.disposed, true);
console.log("cnn: deferred disposal after an in-flight fit passed");

const gan = new GanTrainingSession(tf, cloneTemplate("gan"), {
  batchSize: 16,
  learningRate: 0.001,
});
await gan.step(2);
const ganMetrics = gan.metrics();
assert.ok(Number.isFinite(ganMetrics.trainLoss));
assert.ok(Number.isFinite(ganMetrics.validationLoss));
assert.equal(gan.generatedPoints(8).length, 8);
console.log(
  `gan: generator loss ${ganMetrics.trainLoss.toFixed(4)}, ` +
    `discriminator loss ${ganMetrics.validationLoss.toFixed(4)}`,
);
gan.dispose();
