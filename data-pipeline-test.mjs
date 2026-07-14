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
  signalColumns: ["signal_a", "signal_b"],
  timestampColumn: "",
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
assert.equal(
  tabular.summary.splitCounts.train +
    tabular.summary.splitCounts.validation +
    tabular.summary.splitCounts.test,
  tabular.points.length,
);
assert.equal(tabular.pointSplits.train.length, tabular.summary.splitCounts.train);

const leakageRows = ["a,b,c,target"];
for (let index = 0; index < 12; index += 1) {
  const outlier = index >= 10 ? 1000 + index : index;
  leakageRows.push(`${outlier},${index * 2},${index % 3},${index % 2 ? "yes" : "no"}`);
}
const leakageSafe = prepareTabularDataset(parseCsvText(leakageRows.join("\n")), {
  featureColumns: ["a", "b", "c"],
  targetColumn: "target",
  scaling: "standard",
  split: {
    strategy: "chronological",
    trainRatio: 0.7,
    validationRatio: 0.15,
    testRatio: 0.15,
    seed: 4,
  },
});
assert.equal(leakageSafe.inputShape[0], 3);
assert.ok(leakageSafe.preprocessor.stats[0].maximum < 1000);
assert.equal(leakageSafe.visualization.baseline.length, 3);

const rows = ["step,value,label"];
for (let index = 0; index < 40; index += 1) {
  rows.push(`${index},${Math.sin(index / 4).toFixed(5)},${index < 20 ? "down" : "up"}`);
}
const reversedTimeRows = [rows[0], ...rows.slice(1).reverse()];
const timeSeries = prepareTimeSeriesDataset(parseCsvText(reversedTimeRows.join("\n")), {
  signalColumns: ["value"],
  targetColumn: "label",
  timestampColumn: "step",
  sequenceLength: 8,
  stride: 2,
});
assert.deepEqual(timeSeries.inputShape, [8, 1]);
assert.equal(
  timeSeries.datasetData.splits.train.xs.length,
  timeSeries.datasetData.splits.train.count * 8,
);
assert.ok(timeSeries.datasetData.splits.validation.count > 0);
assert.ok(timeSeries.datasetData.splits.test.count > 0);
assert.equal(timeSeries.datasetData.splitStrategy, "chronological");
assert.equal(timeSeries.summary.timestamp, "step");

const scaled = scaleMatrix([[1, 10], [2, 20], [3, 30]], "minmax");
assert.deepEqual(scaled.matrix[0], [-1, -1]);
assert.deepEqual(scaled.matrix[2], [1, 1]);

const imageSummary = summarizeImageFolder([
  { type: "image/png", name: "a.png", webkitRelativePath: "dataset/cat/a.png" },
  { type: "image/png", name: "b.png", webkitRelativePath: "dataset/dog/b.png" },
]);
assert.deepEqual(imageSummary, {
  imageCount: 2,
  classes: ["cat", "dog"],
  classCounts: { cat: 1, dog: 1 },
});

console.log(
  `data pipeline: ${tabular.points.length} tabular rows, ` +
    `${timeSeries.datasetData.count} sequence windows, ` +
    `${imageSummary.classes.length} image classes`,
);
