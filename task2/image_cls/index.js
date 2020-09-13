import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from './imagenet_classes';

const MODEL_PATH = 'https://raw.githubusercontent.com/paulsp94/tfjs_resnet_imagenet/master/ResNet101/model.json';

const IMAGE_SIZE = 224;

let resnet;
const LoadModel = async () => {
  status('Loading model...');

  resnet = await tf.loadLayersModel(MODEL_PATH);
  resnet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  status('');
  document.getElementById('file-container').style.display = '';
};

async function predict(imgElement) {
  status('Predicting...');

  const startTime1 = performance.now();
  let startTime2;
  const logits = tf.tidy(() => {
    const img = tf.browser.fromPixels(imgElement).toFloat();
    const batched = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    startTime2 = performance.now();

    return resnet.predict(batched);
  });


  const classes = await getTop3Classes(logits);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
  showResults(imgElement, classes);
}

export async function getTop3Classes(logits) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(3);
  const topkIndices = new Int32Array(3);
  for (let i = 0; i < 3; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}


function showResults(imgElement, classes) {
  var rel = "";
  for (let i = 0; i < classes.length; i++) {
    rel += classes[i].className;
    rel += "\n";
  }
  const rel_container = document.getElementById('result');
  rel_container.innerText = rel;
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  for (let i = 0, f; f = files[i]; i++) {
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      var img = document.getElementById("_img");  
      img.src = e.target.result;
      img.onload = () => predict(img);
    };
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;


LoadModel();