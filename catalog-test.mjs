import assert from "node:assert/strict";
import * as tf from "@tensorflow/tfjs";
import {
  ACTIVATIONS,
  MODEL_FAMILIES,
  cloneTemplate,
  createLayer,
  validateArchitecture,
} from "./layer-catalog.mjs";
import {
  buildClassifier,
  buildGan,
  countParameters,
} from "./tf-models.mjs";

const expectedShapes = {
  mlp: [8],
  cnn: [12],
  rnn: [8],
};

for (const family of ["mlp", "cnn", "rnn"]) {
  const layers = cloneTemplate(family);
  const validation = validateArchitecture(MODEL_FAMILIES[family].inputShape, layers);
  assert.equal(validation.valid, true, `${family} template shape should be valid`);
  assert.deepEqual(validation.outputShape, expectedShapes[family]);

  const model = buildClassifier(tf, family, layers);
  const input = tf.zeros([1, ...MODEL_FAMILIES[family].inputShape]);
  const output = model.predict(input);
  assert.deepEqual(output.shape, [1, 1]);
  assert.ok(countParameters(model) > 0);
  console.log(`${family}: output ${output.shape.join("x")}, params ${countParameters(model)}`);
  input.dispose();
  output.dispose();
  model.dispose();
}

for (const activation of ACTIVATIONS) {
  const layers = [createLayer("dense", { units: 4, activation: activation.value })];
  const model = buildClassifier(tf, "mlp", layers);
  const output = model.predict(tf.zeros([1, 2]));
  assert.deepEqual(output.shape, [1, 1], `${activation.value} should build`);
  output.dispose();
  model.dispose();
}
console.log(`${ACTIVATIONS.length} activation functions compiled`);

const ganDefinition = cloneTemplate("gan");
const gan = buildGan(tf, ganDefinition);
const noise = tf.zeros([2, 4]);
const generated = gan.generator.predict(noise);
const judged = gan.discriminator.predict(generated);
assert.deepEqual(generated.shape, [2, 2]);
assert.deepEqual(judged.shape, [2, 1]);
console.log(
  `gan: generator params ${gan.generator.countParams()}, ` +
    `discriminator params ${gan.discriminator.countParams()}`,
);
noise.dispose();
generated.dispose();
judged.dispose();
gan.generator.dispose();
gan.discriminator.dispose();
