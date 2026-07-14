import { parse } from "./node_modules/csv-parse/dist/esm/sync.js";

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

function fillMissingColumns(rows, columns, strategy) {
  const means = new Map();
  if (strategy === "mean") {
    for (const column of columns) {
      const values = rows
        .map((row) => numericValue(row[column]))
        .filter(Number.isFinite);
      if (!values.length) throw new Error(`Column "${column}" has no numeric values`);
      means.set(column, values.reduce((sum, value) => sum + value, 0) / values.length);
    }
  }

  return rows.flatMap((row) => {
    const values = columns.map((column) => numericValue(row[column]));
    if (values.every(Number.isFinite)) return [{ row, values }];
    if (strategy === "mean") {
      return [{
        row,
        values: values.map((value, index) =>
          Number.isFinite(value) ? value : means.get(columns[index])),
      }];
    }
    return [];
  });
}

export function scaleMatrix(matrix, strategy = "standard") {
  if (!matrix.length || !matrix[0]?.length) {
    throw new Error("Scaling requires at least one row and one feature");
  }
  if (strategy === "none") {
    return {
      matrix: matrix.map((row) => row.slice()),
      stats: matrix[0].map(() => ({ strategy: "none" })),
    };
  }

  const featureCount = matrix[0].length;
  const stats = [];
  for (let column = 0; column < featureCount; column += 1) {
    const values = matrix.map((row) => row[column]);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance =
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    stats.push({
      strategy,
      minimum,
      maximum,
      mean,
      standardDeviation: Math.sqrt(variance),
    });
  }

  return {
    matrix: matrix.map((row) =>
      row.map((value, column) => {
        const stat = stats[column];
        if (strategy === "minmax") {
          const range = stat.maximum - stat.minimum;
          return range > 1e-12 ? (value - stat.minimum) / range * 2 - 1 : 0;
        }
        return stat.standardDeviation > 1e-12
          ? (value - stat.mean) / stat.standardDeviation
          : 0;
      })),
    stats,
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
  let targetIndex = normalized.findIndex((header) => targetNames.includes(header));
  if (targetIndex < 0) targetIndex = headers.length - 1;
  const featureColumns = headers.filter((_, index) => index !== targetIndex);
  return {
    targetColumn: headers[targetIndex],
    featureColumns: featureColumns.slice(0, 2),
    signalColumn: featureColumns[0],
  };
}

export function prepareTabularDataset(parsed, options) {
  const {
    featureColumns,
    targetColumn,
    missingStrategy = "drop",
    scaling = "standard",
  } = options;
  if (featureColumns.length !== 2) {
    throw new Error("MLP visualization currently requires exactly two feature columns");
  }
  const validTargets = parsed.rows.filter((row) =>
    row[targetColumn] !== undefined && String(row[targetColumn]).trim() !== "");
  const labelEncoder = binaryLabels(validTargets.map((row) => row[targetColumn]));
  const completed = fillMissingColumns(validTargets, featureColumns, missingStrategy);
  if (completed.length < 8) throw new Error("At least 8 valid rows are required");
  const scaled = scaleMatrix(completed.map((item) => item.values), scaling);
  const points = completed.map((item, index) => ({
    input: scaled.matrix[index],
    target: labelEncoder.encode(item.row[targetColumn]),
  }));
  return {
    family: "mlp",
    points,
    inputShape: [2],
    summary: {
      rows: points.length,
      features: featureColumns.slice(),
      target: targetColumn,
      classes: labelEncoder.classes,
      droppedRows: parsed.rows.length - points.length,
      scaling,
    },
  };
}

export function prepareTimeSeriesDataset(parsed, options) {
  const {
    signalColumn,
    targetColumn,
    sequenceLength = 12,
    stride = 1,
    missingStrategy = "drop",
    scaling = "standard",
  } = options;
  const validTargets = parsed.rows.filter((row) =>
    row[targetColumn] !== undefined && String(row[targetColumn]).trim() !== "");
  const labelEncoder = binaryLabels(validTargets.map((row) => row[targetColumn]));
  const completed = fillMissingColumns(validTargets, [signalColumn], missingStrategy);
  if (completed.length < sequenceLength + 1) {
    throw new Error(`Time series requires more than ${sequenceLength} valid rows`);
  }
  const scaled = scaleMatrix(completed.map((item) => item.values), scaling).matrix;
  const windows = [];
  const labels = [];
  for (
    let start = 0;
    start + sequenceLength <= completed.length;
    start += Math.max(1, stride)
  ) {
    windows.push(scaled.slice(start, start + sequenceLength).map((row) => row[0]));
    const targetRow = completed[start + sequenceLength - 1].row;
    labels.push(labelEncoder.encode(targetRow[targetColumn]));
  }
  if (windows.length < 8) throw new Error("At least 8 sequence windows are required");
  return {
    family: "rnn",
    datasetData: {
      count: windows.length,
      inputShape: [sequenceLength, 1],
      xs: Float32Array.from(windows.flat()),
      ys: Float32Array.from(labels),
    },
    inputShape: [sequenceLength, 1],
    summary: {
      rows: completed.length,
      windows: windows.length,
      sequenceLength,
      stride: Math.max(1, stride),
      signal: signalColumn,
      target: targetColumn,
      classes: labelEncoder.classes,
      scaling,
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
  return { imageCount: images.length, classes };
}

export async function prepareImageDataset(files, options = {}) {
  const imageSize = Number(options.imageSize ?? 8);
  const pixelScaling = options.pixelScaling ?? "zeroOne";
  const images = [...files].filter((file) => file.type.startsWith("image/"));
  if (images.length < 8) throw new Error("At least 8 image files are required");
  const labelEncoder = binaryLabels(images.map(imageLabel));
  const values = new Float32Array(images.length * imageSize * imageSize);
  const labels = new Float32Array(images.length);
  const canvas = document.createElement("canvas");
  canvas.width = imageSize;
  canvas.height = imageSize;
  const context = canvas.getContext("2d", { willReadFrequently: true });

  for (let index = 0; index < images.length; index += 1) {
    const bitmap = await createImageBitmap(images[index]);
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
    labels[index] = labelEncoder.encode(imageLabel(images[index]));
  }

  return {
    family: "cnn",
    datasetData: {
      count: images.length,
      inputShape: [imageSize, imageSize, 1],
      xs: values,
      ys: labels,
    },
    inputShape: [imageSize, imageSize, 1],
    summary: {
      images: images.length,
      imageSize,
      channels: 1,
      classes: labelEncoder.classes,
      pixelScaling,
    },
  };
}
