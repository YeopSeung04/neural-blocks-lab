import assert from "node:assert/strict";
import {
  parseCsvText,
  prepareTabularDataset,
  prepareTimeSeriesDataset,
  scaleMatrix,
  suggestColumnMapping,
  summarizeImageFolder,
} from "./data-pipeline.mjs";

const tabularText = [
  "signal_a,signal_b,target",
  "-2,-1,no",
  "-1,-2,no",
  "-0.5,-1,no",
  "0.2,0.5,yes",
  "0.7,0.4,yes",
  "1,1.3,yes",
  "1.2,0.8,yes",
  "-1.5,-0.8,no",
  ",0.9,yes",
].join("\n");
const parsedTabular = parseCsvText(tabularText);
assert.deepEqual(suggestColumnMapping(parsedTabular.headers), {
  targetColumn: "target",
  featureColumns: ["signal_a", "signal_b"],
  signalColumn: "signal_a",
});
const tabular = prepareTabularDataset(parsedTabular, {
  featureColumns: ["signal_a", "signal_b"],
  targetColumn: "target",
  missingStrategy: "mean",
  scaling: "standard",
});
assert.equal(tabular.points.length, 9);
assert.equal(tabular.points[0].input.length, 2);
assert.deepEqual(tabular.summary.classes, ["no", "yes"]);

const rows = ["value,label"];
for (let index = 0; index < 40; index += 1) {
  rows.push(`${Math.sin(index / 4).toFixed(5)},${index < 20 ? "down" : "up"}`);
}
const timeSeries = prepareTimeSeriesDataset(parseCsvText(rows.join("\n")), {
  signalColumn: "value",
  targetColumn: "label",
  sequenceLength: 8,
  stride: 2,
});
assert.deepEqual(timeSeries.inputShape, [8, 1]);
assert.equal(timeSeries.datasetData.xs.length, timeSeries.datasetData.count * 8);
assert.equal(timeSeries.datasetData.ys.length, timeSeries.datasetData.count);

const scaled = scaleMatrix([[1, 10], [2, 20], [3, 30]], "minmax");
assert.deepEqual(scaled.matrix[0], [-1, -1]);
assert.deepEqual(scaled.matrix[2], [1, 1]);

const imageSummary = summarizeImageFolder([
  { type: "image/png", name: "a.png", webkitRelativePath: "dataset/cat/a.png" },
  { type: "image/png", name: "b.png", webkitRelativePath: "dataset/dog/b.png" },
]);
assert.deepEqual(imageSummary, { imageCount: 2, classes: ["cat", "dog"] });

console.log(
  `data pipeline: ${tabular.points.length} tabular rows, ` +
    `${timeSeries.datasetData.count} sequence windows, ` +
    `${imageSummary.classes.length} image classes`,
);
