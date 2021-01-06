
 window.addEventListener("load", () => {
    const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");

    //canvas.height = window.innerHeight;
    //canvas.window = window.innerWidth;
canvas.height = 100;
canvas.window = 100;
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

      ctx.lineTo(e.clientX, e.clientY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.clientX, e.clientY);
    }



    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishedPosition);
    canvas.addEventListener("mousemove", draw);



    })

function myFunction(){
      const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");

    //canvas.height = window.innerHeight;
    //canvas.window = window.innerWidth;
canvas.height = 100;
canvas.window = 100;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

}


function applyPreprocessing(img){


//  var img = document.getElementById('my-image').asType('float32');
 
  const canvas = document.querySelector("#canvas");
  var img = tf.browser.fromPixels(canvas);
    //cast to dtype float32
  img = img.asType('float32');
  

  //greyscale
  // reshape to dim  [1, width, height, 1]
   img = tf.browser.fromPixels(img)
    .mean(2)
    .toFloat()
    .expandDims(0)
    .expandDims(-1);

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
