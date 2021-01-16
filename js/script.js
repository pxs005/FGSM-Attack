
let model;
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log(model.summary());
    console.log("Finished loading model");
};

loadModel();

window.addEventListener("load", () => {
    const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");

    //canvas.height = window.innerHeight;
    //canvas.window = window.innerWidth;
    canvas.height = 300;
    canvas.window = 300;
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

        ctx.lineWidth = 15;
        ctx.lineCap = "round";
        //ctx.strokeStyle = "#FF0000";

        ctx.lineTo(e.clientX, e.clientY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(e.clientX, e.clientY);
    }

    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishedPosition);
    canvas.addEventListener("mousemove", draw);
});

function erasePad() {
    const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");

    //canvas.height = window.innerHeight;
    //canvas.window = window.innerWidth;
    canvas.height = 300;
    canvas.window = 300;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    let painting = false;

}


function handleImageStuff() {
    var img = applyPreprocessing(); //apply preprocessing
    var pred = predictOriginal(model, img);
    pred.print();

    var x = tf.squeeze(pred);

    x = x.argMax(); //prediced value of original image
    console.log(typeof x);
    console.log("Predicted value:");
    x.print();


    //display the original image and predicted value and model's confidence
    displayOriginalImage(img);
    //document.getElementById('original image').innerHTML = img;
    document.getElementById('original prediction').innerHTML = '<p>Predicted Value: </p>' + x;
    //Retrieve the image label and epsilon value and store in an array
    var arr = [];
    arr = retrieveVals(arr);

    //Need to be fixed! Need to address the issue of the loss function
    var gradient = getGradient(img, arr[0], pred); //need to fix
    var advImage = generateAdversarialImage(img, arr[1], gradient);

    var advPred = predictAdversarial(model, img);
    var y = tf.squeeze(advPred);
    y = y.argMax(); //predicted value of adversarial image

    console.log("Predicted Adversarial value:");
    y.print();

    //display the adversarial image and predicted value and model's confidence
    //displayAdversarialImage(advImage);

}


function retrieveVals(arr) {
    var x = document.getElementById("myForm");
    var i;

    for(i = 0; i < x.length; i++) {
        var text = x.elements[i].value;
        console.log(typeof text);
        arr.push(text);
    }

    return arr;

}



function displayOriginalImage(parm) {
    parm = tf.image.resizeNearestNeighbor(parm, [100, 100]);
    var parm = tf.squeeze(parm);
    var x = parm.shape;
    tf.browser.toPixels(parm, document.getElementsByTagName('canvas')[1]);
    //tf.browser.toPixels(parm, document.getElementsByTagName("original image")[0]);
}

function predictOriginal(model, preprocessed_image) {
    result = model.predict(preprocessed_image);
    return result;
}


function displayAdversarialImage(adv_image) {
    //maybe display the image in a canvas element
    //tf.browser.toPixels(adv_image, document.getElementsByTagName("canvas2")[0]);
}

function predictAdversarial(model, adv_image) {
    result = model.predict(adv_image);
    return result;
}

function applyPreprocessing() {
    const canvas = document.querySelector("#canvas");
    var img = tf.browser.fromPixels(canvas);

    img = tf.image.resizeNearestNeighbor(img, [28, 28]);
    img = img.mean(2).toFloat().expandDims(0).expandDims(-1); //Shape: [1, 28,28,1]
    img = tf.div(img,255);
    console.log("Image's shape",img.shape);

    return img
}


function calculateLoss(yTrue, yPred) {
    yTrue = tf.oneHot(tf.tensor1d([yTrue], 'int32'), 10);
    var x = tf.metrics.categoricalCrossentropy(yTrue,yPred);
    return x;
}

function getGradient(img, yTrue, yPred) {
    /*
    const g = tf.grad(calculateLoss(yTrue,yPred));
    return g;
    */
    function f(img) {
        yTrue = tf.oneHot(tf.tensor1d([yTrue], 'int32'), 10);
        console.log(yTrue.shape);
        return tf.metrics.categoricalCrossentropy(yTrue, yPred);
        
    }
    img = tf.image.resizeNearestNeighbor(img, [28, 28]);
    img = tf.squeeze(img);
    console.log(img.shape);
    var g = tf.grad(f);
    g(img).print();
   
}

/*
function f(img) {
    yTrue = tf.oneHot(tf.tensor1d([yTrue], 'int32'), 10);
    return tf.metrics.categoricalCrossentropy(model.predict(img), lbl);
    // (Typo: the order of arguments should be flipped, but it does not affect the question here)
}
var g = tf.grad(f);
g(img).print();
*/
function generateAdversarialImage(image,epsilon,pertubation) {
    //pertubation from getGradient function
    var x = tf.sign(pertubation).mul(epsilon);
    x = image.add(x).clipByValue(0,1);

    //  image + epsilon * pertubation;
    return x;
}

//might get rid
function retrievePredictionDict(model, img){

  dict = {};
  var label = model.predict(img);

  dict.append(label);

  //also get prediction of confidence

  //append to dict
  return dict


}
