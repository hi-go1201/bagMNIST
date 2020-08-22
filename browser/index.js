async function runExample() {
  // Create an ONNX inference session with WebGL backend.
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });

  // Load an ONNX model. This model is Resnet50 that takes a 1*3*224*224 image and classifies it.
  await session.loadModel("./bagMNIST.onnx");

  //opencv.jsで画像を28x28背景黒のグレースケール画像に変換
  //dst2の手法ならcanvasへの描画も一つでできるが、精度が悪い。dstだけだとなぜかcanvasに表示されない
  //let src = cv.matFromImageData(imageData);
  let imgElement = document.getElementById('imageSource');
  let src = cv.imread(imgElement);
  let dst = new cv.Mat();
  let dsize = new cv.Size(imageSize, imageSize);
  src.convertTo(dst, cv.CV_8U);
  // You can try more different parameters
  cv.resize(dst, dst, dsize, 0, 0, cv.INTER_AREA);
  cv.bitwise_not(dst, dst);
  const testData = new ImageData(new Uint8ClampedArray(dst.data, dst.cols, dst.rows), imageSize, imageSize);

  let dst2 = new cv.Mat();
  cv.cvtColor(src, dst2, cv.COLOR_RGBA2GRAY, 0); 
  cv.resize(dst2, dst2, dsize, 0, 0, cv.INTER_AREA);
  cv.bitwise_not(dst2, dst2);
  cv.cvtColor(dst2, dst2, cv.COLOR_GRAY2RGBA, 4);
  //const testData2 = new ImageData(new Uint8ClampedArray(dst2.data, dst2.cols, dst2.rows), imageSize, imageSize);
  cv.imshow('testCanvas', dst2);
  src.delete();
  dst.delete();
  dst2.delete();

  //ToDo: CGのテクスチャ用にカバンの正面を領域抽出してクロップできるか検証

  // Preprocess the image data to match input dimension requirement, which is 1*3*224*224.
  const width = imageSize;
  const height = imageSize;
  //const preprocessedData = preprocess(imageData.data, width, height);
  const preprocessedData = preprocess(testData.data, width, height);
  
  const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 1, width, height]);
  // Run model with Tensor inputs and get the result.
  const outputMap = await session.run([inputTensor]);
  const outputData = outputMap.values().next().value.data;

  const maxPrediction = Math.max(...outputData);

  console.log(outputData.length);
  console.log(outputData);
  console.log(maxPrediction);

  const element = document.getElementById('predictions');
  element.innerHTML = '';
  const results = [];
  for (let i = 0; i < outputData.length; i++) {
    results.push(`${bagMNISTClasses[i][0]}: ${outputData[i] * 100}%`);
  }  
  element.innerHTML = results.join('<br/>');

  // Render the output result in html.
  // printMatches(outputData);
}

var bagMNISTClasses = {
  '0': ['Brief/Tote'],
  '1': ['Shoulder/Messenger'],
  '2': ['Backpack'],
  '3': ['Boston'],
  '4': ['Body/Clutch'],
  '5': ['Other'],
};

/**
 * Preprocess raw image data to match Resnet50 requirement.
 */
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 1), [1, 1, height, width]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 128.0);
  ndarray.ops.subseq(dataFromImage, 1.0);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

  return dataProcessed.data;
}