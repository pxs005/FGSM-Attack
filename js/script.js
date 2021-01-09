
 let model;
async function loadModel() {
//   model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
  


  console.log("Finished loading model");

};

loadModel();
 

window.addEventListener("load", () => {
    const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");

    //canvas.height = window.innerHeight;
    //canvas.window = window.innerWidth;
canvas.height = 100;
canvas.window = 100;
       ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    let painting = false;


    function startPosition(e) {
      painting = true;
      draw(e);
    }

    function finishedPosition() {
      painting = false;
      ctx.beginPath();
    }


    function draw(e) {
      if (!painting) return;

      ctx.linewWidth = 60;
      ctx.lineCap = "round";
     // ctx.strokeStyle = "#FF0000";

      ctx.lineTo(e.clientX, e.clientY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.clientX, e.clientY);
    }



    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishedPosition);
    canvas.addEventListener("mousemove", draw);



    });

function erasePad(){
      const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");

    //canvas.height = window.innerHeight;
    //canvas.window = window.innerWidth;
canvas.height = 100;
canvas.window = 100;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

}


function handleImageStuff(){
  //apply preprocessing
  var img = applyPreprocessing(); 
  //applyPreprocessing();
  displayOriginalImage(img);

  
}


function displayOriginalImage(parm){
  
  
tf.browser.toPixels(parm, document.getElementsByTagName("canvas")[0]);

}

function predictOriginal(){
  
}
function displayAdversarialImage(){
  
}





function applyPreprocessing(){
 
  const canvas = document.querySelector("#canvas");
  var img = tf.browser.fromPixels(canvas);

  img = tf.image.resizeNearestNeighbor(img, [28, 28]);
  img = img.mean(2);

  
  img = tf.reshape(img, [1, 28*28]);


    //cast to dtype float32
  //img = img.asType('float32');
  

  //greyscale
  // reshape to dim  [1, width, height, 1]

    // img = img.mean(2).toFloat().expandDims(0).expandDims(-1); 

  return img


}


function calculateLoss(img, lbl){
  var x = tf.metrics.categoricalCrossentropy(model.predict(img), lbl);
  return x;
}

function getGradient(img, label){
  const g = tf.grad(calculateLoss(img,label));
  return g;
}

function generateAdversarialImage(image,epsilon,pertubation){

  var x = image + epsilon * pertubation;

  return x;

}

function retrievePredictionDict(model, img){

  dict = {};
  var label = model.predict(img);

  dict.append(label);

  //also get prediction of confidence

  //append to dict
  return dict


}
