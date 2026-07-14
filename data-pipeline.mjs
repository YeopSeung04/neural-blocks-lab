import { parse } from "./node_modules/csv-parse/dist/esm/sync.js";

function createRng(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function shuffled(values, seed) {
  const random = createRng(seed);
  const result = values.slice();
  for (let index = result.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [result[index], result[target]] = [result[target], result[index]];
  }
  return result;
}

function numericValue(value) {
  if (value === null || value === undefined || String(value).trim() === "") return NaN;
  return Number(value);
}

function binaryLabels(values) {
  const classes = [...new Set(values.map((value) => String(value).trim()))].sort();
  if (classes.length !== 2) {
    throw new Error(`Binary classification requires exactly 2 labels; received ${classes.length}`);
  }
  const mapping = new Map(classes.map((label, index) => [label, index]));
  return {
    classes,
    encode: (value) => mapping.get(String(value).trim()),
  };
}

function normalizeSplitConfig(config = {}) {
  const trainRatio = Number(config.trainRatio ?? 0.7);
  const validationRatio = Number(config.validationRatio ?? 0.15);
  const testRatio = Number(config.testRatio ?? 1 - trainRatio - validationRatio);
  const total = trainRatio + validationRatio + testRatio;
  if (
    !Number.isFinite(total) ||
    trainRatio <= 0 ||
    validationRatio < 0 ||
    testRatio < 0 ||
    Math.abs(total - 1) > 1e-6
  ) {
    throw new Error("Train, validation, and test ratios must add up to 1");
  }
  return {
    strategy: config.strategy ?? "stratified",
    trainRatio,
    validationRatio,
    testRatio,
    seed: Number(config.seed ?? 42),
  };
}

function splitCounts(count, config) {
  if (count < 3) throw new Error("At least 3 samples are required for data splitting");
  let validationCount = config.validationRatio === 0
    ? 0
    : Math.max(1, Math.floor(count * config.validationRatio));
  let testCount = config.testRatio === 0
    ? 0
    : Math.max(1, Math.floor(count * config.testRatio));
  if (validationCount + testCount >= count) {
    if (testCount > 0) testCount = Math.max(0, testCount - 1);
    else validationCount = Math.max(0, validationCount - 1);
  }
  const trainCount = count - validationCount - testCount;
  return { trainCount, validationCount, testCount };
}

function randomSplit(items, config) {
  const ordered = config.strategy === "chronological"
    ? items.slice()
    : shuffled(items, config.seed);
  const { trainCount, validationCount } = splitCounts(ordered.length, config);
  return {
    train: ordered.slice(0, trainCount),
    validation: ordered.slice(trainCount, trainCount + validationCount),
    test: ordered.slice(trainCount + validationCount),
  };
}

function stratifiedSplit(items, labelAccessor, config) {
  const groups = new Map();
  for (const item of items) {
    const label = String(labelAccessor(item));
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(item);
  }
  const result = { train: [], validation: [], test: [] };
  let offset = 0;
  for (const group of groups.values()) {
    const split = randomSplit(group, {
      ...config,
      strategy: "random",
      seed: config.seed + offset,
    });
    result.train.push(...split.train);
    result.validation.push(...split.validation);
    result.test.push(...split.test);
    offset += 97;
  }
  return {
    train: shuffled(result.train, config.seed + 11),
    validation: shuffled(result.validation, config.seed + 12),
    test: shuffled(result.test, config.seed + 13),
  };
}

export function splitRecords(items, options = {}) {
  const config = normalizeSplitConfig(options);
  if (config.strategy === "stratified") {
    if (typeof options.labelAccessor !== "function") {
      throw new Error("Stratified split requires a label accessor");
    }
    return stratifiedSplit(items, options.labelAccessor, config);
  }
  return randomSplit(items, config);
}

function fitNumericPreprocessor(rows, columns, missingStrategy, scaling) {
  const means = columns.map((column) => {
    const values = rows
      .map((row) => numericValue(row[column]))
      .filter(Number.isFinite);
    if (!values.length) throw new Error(`Column "${column}" has no numeric train values`);
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  });

  const trainMatrix = rows.flatMap((row) => {
    const values = columns.map((column, index) => {
      const value = numericValue(row[column]);
      return Number.isFinite(value) ? value : means[index];
    });
    if (missingStrategy === "drop") {
      const hasMissing = columns.some((column) => !Number.isFinite(numericValue(row[column])));
      if (hasMissing) return [];
    }
    return [values];
  });
  if (!trainMatrix.length) throw new Error("No train rows remain after missing-value handling");

  const stats = columns.map((_, column) => {
    const values = trainMatrix.map((row) => row[column]);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance =
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    return {
      strategy: scaling,
      minimum,
      maximum,
      mean,
      standardDeviation: Math.sqrt(variance),
    };
  });
  return { columns: columns.slice(), missingStrategy, scaling, means, stats };
}

function transformNumericRows(rows, preprocessor) {
  const transformed = [];
  let droppedRows = 0;
  for (const row of rows) {
    const raw = preprocessor.columns.map((column, index) => {
      const value = numericValue(row[column]);
      return Number.isFinite(value) ? value : preprocessor.means[index];
    });
    if (
      preprocessor.missingStrategy === "drop" &&
      preprocessor.columns.some((column) => !Number.isFinite(numericValue(row[column])))
    ) {
      droppedRows += 1;
      continue;
    }
    const values = raw.map((value, column) => {
      const stat = preprocessor.stats[column];
      if (preprocessor.scaling === "none") return value;
      if (preprocessor.scaling === "minmax") {
        const range = stat.maximum - stat.minimum;
        return range > 1e-12 ? (value - stat.minimum) / range * 2 - 1 : 0;
      }
      return stat.standardDeviation > 1e-12
        ? (value - stat.mean) / stat.standardDeviation
        : 0;
    });
    transformed.push({ row, values });
  }
  return { rows: transformed, droppedRows };
}

export function scaleMatrix(matrix, strategy = "standard") {
  if (!matrix.length || !matrix[0]?.length) {
    throw new Error("Scaling requires at least one row and one feature");
  }
  const columns = matrix[0].map((_, index) => `feature_${index}`);
  const rows = matrix.map((values) =>
    Object.fromEntries(values.map((value, index) => [columns[index], value])));
  const preprocessor = fitNumericPreprocessor(rows, columns, "drop", strategy);
  return {
    matrix: transformNumericRows(rows, preprocessor).rows.map((item) => item.values),
    stats: preprocessor.stats,
  };
}

export function parseCsvText(text) {
  const rows = parse(text, {
    bom: true,
    columns: true,
    delimiter: [",", "\t", ";"],
    relax_column_count: true,
    skip_empty_lines: true,
    trim: true,
  });
  if (!rows.length) throw new Error("CSV contains no data rows");
  const headers = Object.keys(rows[0]);
  if (headers.length < 2) throw new Error("CSV requires at least two columns");
  return { headers, rows };
}

export function suggestColumnMapping(headers) {
  const normalized = headers.map((header) => header.toLowerCase());
  const targetNames = ["target", "label", "class", "y", "result"];
  const timestampNames = ["timestamp", "time", "date", "datetime", "step"];
  let targetIndex = normalized.findIndex((header) => targetNames.includes(header));
  if (targetIndex < 0) targetIndex = headers.length - 1;
  const timestampIndex = normalized.findIndex((header) => timestampNames.includes(header));
  const featureColumns = headers.filter((_, index) =>
    index !== targetIndex && index !== timestampIndex);
  return {
    targetColumn: headers[targetIndex],
    featureColumns,
    signalColumns: featureColumns,
    timestampColumn: timestampIndex >= 0 ? headers[timestampIndex] : "",
  };
}

export function profileTabularData(parsed, options = {}) {
  const columns = options.columns?.length ? options.columns : parsed.headers;
  const targetColumn = options.targetColumn;
  const columnProfiles = columns.map((column) => {
    const rawValues = parsed.rows.map((row) => row[column]);
    const missing = rawValues.filter((value) =>
      value === undefined || String(value).trim() === "").length;
    const numeric = rawValues.map(numericValue).filter(Number.isFinite);
    const unique = new Set(rawValues.map((value) => String(value).trim())).size;
    return {
      column,
      missing,
      unique,
      numeric: numeric.length === rawValues.length - missing,
      minimum: numeric.length ? Math.min(...numeric) : null,
      maximum: numeric.length ? Math.max(...numeric) : null,
    };
  });
  const classCounts = {};
  if (targetColumn) {
    for (const row of parsed.rows) {
      const label = String(row[targetColumn] ?? "").trim();
      if (label) classCounts[label] = (classCounts[label] ?? 0) + 1;
    }
  }
  return {
    rowCount: parsed.rows.length,
    columnCount: parsed.headers.length,
    columns: columnProfiles,
    classCounts,
    preview: parsed.rows.slice(0, 5),
  };
}

function median(values) {
  const sorted = values.slice().sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function mapTabularSplit(rows, preprocessor, labelEncoder, targetColumn) {
  const transformed = transformNumericRows(rows, preprocessor);
  return {
    points: transformed.rows.map((item) => ({
      input: item.values,
      target: labelEncoder.encode(item.row[targetColumn]),
    })),
    droppedRows: transformed.droppedRows,
  };
}

export function prepareTabularDataset(parsed, options) {
  const {
    featureColumns,
    targetColumn,
    missingStrategy = "drop",
    scaling = "standard",
    split = {},
  } = options;
  if (!Array.isArray(featureColumns) || featureColumns.length < 2) {
    throw new Error("MLP training requires at least two feature columns");
  }
  const uniqueFeatures = [...new Set(featureColumns)];
  if (uniqueFeatures.includes(targetColumn)) {
    throw new Error("Target column cannot also be an input feature");
  }
  const validTargets = parsed.rows.filter((row) =>
    row[targetColumn] !== undefined && String(row[targetColumn]).trim() !== "");
  const labelEncoder = binaryLabels(validTargets.map((row) => row[targetColumn]));
  const rawSplits = splitRecords(validTargets, {
    strategy: split.strategy ?? "stratified",
    trainRatio: split.trainRatio,
    validationRatio: split.validationRatio,
    testRatio: split.testRatio,
    seed: split.seed,
    labelAccessor: (row) => row[targetColumn],
  });
  const preprocessor = fitNumericPreprocessor(
    rawSplits.train,
    uniqueFeatures,
    missingStrategy,
    scaling,
  );
  const transformed = {
    train: mapTabularSplit(rawSplits.train, preprocessor, labelEncoder, targetColumn),
    validation: mapTabularSplit(
      rawSplits.validation,
      preprocessor,
      labelEncoder,
      targetColumn,
    ),
    test: mapTabularSplit(rawSplits.test, preprocessor, labelEncoder, targetColumn),
  };
  if (transformed.train.points.length < 4) {
    throw new Error("At least 4 train rows are required after preprocessing");
  }
  const pointSplits = {
    train: transformed.train.points,
    validation: transformed.validation.points,
    test: transformed.test.points,
  };
  const points = [
    ...pointSplits.train,
    ...pointSplits.validation,
    ...pointSplits.test,
  ];
  const baseline = uniqueFeatures.map((_, featureIndex) =>
    median(pointSplits.train.map((point) => point.input[featureIndex])));
  const droppedRows =
    transformed.train.droppedRows +
    transformed.validation.droppedRows +
    transformed.test.droppedRows +
    (parsed.rows.length - validTargets.length);
  return {
    family: "mlp",
    taskType: "binaryClassification",
    points,
    pointSplits,
    inputShape: [uniqueFeatures.length],
    visualization: {
      featureNames: uniqueFeatures.slice(),
      xFeatureIndex: 0,
      yFeatureIndex: 1,
      baseline,
    },
    preprocessor,
    summary: {
      rows: points.length,
      features: uniqueFeatures.slice(),
      target: targetColumn,
      classes: labelEncoder.classes,
      splitCounts: {
        train: pointSplits.train.length,
        validation: pointSplits.validation.length,
        test: pointSplits.test.length,
      },
      droppedRows,
      scaling,
      splitStrategy: split.strategy ?? "stratified",
    },
  };
}

function transformSequenceRows(rows, preprocessor) {
  const transformed = transformNumericRows(rows, preprocessor);
  return {
    rows: transformed.rows.map((item) => ({
      source: item.row,
      values: item.values,
    })),
    droppedRows: transformed.droppedRows,
  };
}

function typedSequenceSplit(windows, labels, sequenceLength, signalCount) {
  return {
    count: windows.length,
    xs: Float32Array.from(windows.flat(2)),
    ys: Float32Array.from(labels),
    inputShape: [sequenceLength, signalCount],
  };
}

export function prepareTimeSeriesDataset(parsed, options) {
  const {
    signalColumns: configuredSignalColumns,
    signalColumn,
    targetColumn,
    timestampColumn = "",
    sequenceLength = 12,
    stride = 1,
    missingStrategy = "drop",
    scaling = "standard",
    split = {},
  } = options;
  const signalColumns = configuredSignalColumns?.length
    ? [...new Set(configuredSignalColumns)]
    : [signalColumn].filter(Boolean);
  if (!signalColumns.length) throw new Error("Select at least one signal column");
  if (signalColumns.includes(targetColumn)) {
    throw new Error("Target column cannot also be a signal");
  }
  const validTargets = parsed.rows.filter((row) =>
    row[targetColumn] !== undefined && String(row[targetColumn]).trim() !== "");
  if (timestampColumn) {
    validTargets.sort((left, right) => {
      const leftNumber = numericValue(left[timestampColumn]);
      const rightNumber = numericValue(right[timestampColumn]);
      if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
        return leftNumber - rightNumber;
      }
      const leftTime = Date.parse(left[timestampColumn]);
      const rightTime = Date.parse(right[timestampColumn]);
      if (Number.isFinite(leftTime) && Number.isFinite(rightTime)) return leftTime - rightTime;
      return String(left[timestampColumn]).localeCompare(String(right[timestampColumn]));
    });
  }
  const usableTargets = missingStrategy === "drop"
    ? validTargets.filter((row) =>
        signalColumns.every((column) => Number.isFinite(numericValue(row[column]))))
    : validTargets;
  const droppedMissingRows = validTargets.length - usableTargets.length;
  const labelEncoder = binaryLabels(usableTargets.map((row) => row[targetColumn]));
  const splitConfig = normalizeSplitConfig({ ...split, strategy: "chronological" });
  const { trainCount, validationCount } = splitCounts(usableTargets.length, splitConfig);
  const trainEnd = trainCount;
  const validationEnd = trainCount + validationCount;
  const preprocessor = fitNumericPreprocessor(
    usableTargets.slice(0, trainEnd),
    signalColumns,
    missingStrategy,
    scaling,
  );
  const transformed = transformSequenceRows(usableTargets, preprocessor);
  if (transformed.rows.length < sequenceLength + 2) {
    throw new Error(`Time series requires more than ${sequenceLength + 1} valid rows`);
  }

  const windows = { train: [], validation: [], test: [] };
  const labels = { train: [], validation: [], test: [] };
  const safeStride = Math.max(1, Number(stride));
  for (
    let start = 0;
    start + sequenceLength <= transformed.rows.length;
    start += safeStride
  ) {
    const end = start + sequenceLength - 1;
    const splitName = end < trainEnd
      ? "train"
      : end < validationEnd
        ? "validation"
        : "test";
    windows[splitName].push(
      transformed.rows.slice(start, start + sequenceLength).map((row) => row.values),
    );
    labels[splitName].push(
      labelEncoder.encode(transformed.rows[end].source[targetColumn]),
    );
  }
  if (windows.train.length < 4) {
    throw new Error("At least 4 train sequence windows are required");
  }
  const signalCount = signalColumns.length;
  const tensorSplits = {
    train: typedSequenceSplit(windows.train, labels.train, sequenceLength, signalCount),
    validation: typedSequenceSplit(
      windows.validation,
      labels.validation,
      sequenceLength,
      signalCount,
    ),
    test: typedSequenceSplit(windows.test, labels.test, sequenceLength, signalCount),
  };
  const count =
    tensorSplits.train.count +
    tensorSplits.validation.count +
    tensorSplits.test.count;
  return {
    family: "rnn",
    taskType: "binaryClassification",
    datasetData: {
      count,
      inputShape: [sequenceLength, signalCount],
      splits: tensorSplits,
      splitStrategy: "chronological",
    },
    inputShape: [sequenceLength, signalCount],
    preprocessor,
    summary: {
      rows: transformed.rows.length,
      windows: count,
      sequenceLength,
      stride: safeStride,
      signals: signalColumns,
      timestamp: timestampColumn || null,
      target: targetColumn,
      classes: labelEncoder.classes,
      splitCounts: {
        train: tensorSplits.train.count,
        validation: tensorSplits.validation.count,
        test: tensorSplits.test.count,
      },
      droppedRows:
        transformed.droppedRows +
        droppedMissingRows +
        (parsed.rows.length - validTargets.length),
      scaling,
      splitStrategy: "chronological",
    },
  };
}

function imageLabel(file) {
  const path = file.webkitRelativePath || file.name;
  const parts = path.split("/").filter(Boolean);
  if (parts.length < 3) {
    throw new Error("Image files must be organized as root/class_name/image.ext");
  }
  return parts[1];
}

export function summarizeImageFolder(files) {
  const images = [...files].filter((file) => file.type.startsWith("image/"));
  const labels = images.map(imageLabel);
  const classes = [...new Set(labels)].sort();
  const classCounts = Object.fromEntries(
    classes.map((label) => [label, labels.filter((value) => value === label).length]),
  );
  return { imageCount: images.length, classes, classCounts };
}

async function tensorizeImages(files, labelEncoder, imageSize, pixelScaling) {
  const values = new Float32Array(files.length * imageSize * imageSize);
  const labels = new Float32Array(files.length);
  const canvas = document.createElement("canvas");
  canvas.width = imageSize;
  canvas.height = imageSize;
  const context = canvas.getContext("2d", { willReadFrequently: true });

  for (let index = 0; index < files.length; index += 1) {
    const bitmap = await createImageBitmap(files[index]);
    context.clearRect(0, 0, imageSize, imageSize);
    context.drawImage(bitmap, 0, 0, imageSize, imageSize);
    bitmap.close();
    const rgba = context.getImageData(0, 0, imageSize, imageSize).data;
    for (let pixel = 0; pixel < imageSize * imageSize; pixel += 1) {
      const offset = pixel * 4;
      const grayscale =
        (rgba[offset] * 0.299 + rgba[offset + 1] * 0.587 + rgba[offset + 2] * 0.114) /
        255;
      values[index * imageSize * imageSize + pixel] =
        pixelScaling === "minusOneOne" ? grayscale * 2 - 1 : grayscale;
    }
    labels[index] = labelEncoder.encode(imageLabel(files[index]));
  }
  return {
    count: files.length,
    inputShape: [imageSize, imageSize, 1],
    xs: values,
    ys: labels,
  };
}

export async function prepareImageDataset(files, options = {}) {
  const imageSize = Number(options.imageSize ?? 8);
  const pixelScaling = options.pixelScaling ?? "zeroOne";
  const images = [...files].filter((file) => file.type.startsWith("image/"));
  if (images.length < 8) throw new Error("At least 8 image files are required");
  const labelEncoder = binaryLabels(images.map(imageLabel));
  const fileSplits = splitRecords(images, {
    strategy: "stratified",
    trainRatio: options.split?.trainRatio,
    validationRatio: options.split?.validationRatio,
    testRatio: options.split?.testRatio,
    seed: options.split?.seed,
    labelAccessor: imageLabel,
  });
  const tensorSplits = {
    train: await tensorizeImages(
      fileSplits.train,
      labelEncoder,
      imageSize,
      pixelScaling,
    ),
    validation: await tensorizeImages(
      fileSplits.validation,
      labelEncoder,
      imageSize,
      pixelScaling,
    ),
    test: await tensorizeImages(
      fileSplits.test,
      labelEncoder,
      imageSize,
      pixelScaling,
    ),
  };
  return {
    family: "cnn",
    taskType: "binaryClassification",
    datasetData: {
      count: images.length,
      inputShape: [imageSize, imageSize, 1],
      splits: tensorSplits,
      splitStrategy: "stratified",
    },
    inputShape: [imageSize, imageSize, 1],
    summary: {
      images: images.length,
      imageSize,
      channels: 1,
      classes: labelEncoder.classes,
      classCounts: summarizeImageFolder(images).classCounts,
      splitCounts: {
        train: tensorSplits.train.count,
        validation: tensorSplits.validation.count,
        test: tensorSplits.test.count,
      },
      pixelScaling,
      splitStrategy: "stratified",
    },
  };
}
