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

    canvas.height = 300;
    canvas.window = 300;
    ctx.fillStyle = 'black';
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
        ctx.strokeStyle = "white";

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

    canvas.height = 300;
    canvas.window = 300;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    let painting = false;
}

function handleImageStuff() {
    var img = applyPreprocessing(); //apply preprocessing
    var pred = predictOriginal(model, img);
    pred.print();

    var x = tf.squeeze(pred);
    console.log("Original Predictions:")
    x.print(); // print probabilities of selected digit
    
    var probArr = x.dataSync();

    x = x.argMax(); //prediced value of original image
    console.log(typeof x);
    x = x.dataSync()[0]; // extract data and store in element
    console.log("Predicted value:", x);

    //display the original image and predicted value and model's confidence
    displayOriginalImage(img);
    document.getElementById('original title').innerHTML = '<p>Original Image</p>';

    var predProb = probArr[x].toFixed(4) *100;
    predProb = Math.round(predProb*100) / 100;
    document.getElementById('original prediction').innerHTML = '<p>Predicted to be a <strong>' + x + '</strong> with <strong>' + predProb + '%</strong> confidence</p>';

    //Retrieve the image label and epsilon value and store in an array
    var arr = [];
    arr = retrieveVals(arr);

    var gradient = getGradient(img, arr[0]); //need to fix
    console.log(gradient.dataSync());
    var advImage = generateAdversarialImage(img, arr[1], gradient);

    var advPred = predictAdversarial(model, advImage);
    var y = tf.squeeze(advPred);

    console.log("Adversarial Predictions:")
    y.print();

    var probAdvArr = y.dataSync();
    y = y.argMax(); //predicted value of adversarial image

    console.log("Predicted Adversarial value:");
    y.print();

    y = y.dataSync()[0]; // extract data and store in element
    console.log("Predicted value:", y);

    //display the adversarial image and predicted value and model's confidence
    displayAdversarialImage(advImage);

    //document.getElementById('original image').innerHTML = img;
    document.getElementById('adversarial title').innerHTML = '<p>Adversarial Image</p>';

    var predAdvProb = probAdvArr[y].toFixed(4) *100;
    predAdvProb = Math.round(predAdvProb*100) / 100;
    document.getElementById('adversarial prediction').innerHTML = '<p>Predicted to be a <strong>' + y + '</strong> with <strong>' + predAdvProb + '%</strong> confidence</p>';
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
    parm = tf.image.resizeNearestNeighbor(parm, [300, 300]);
    var parm = tf.squeeze(parm);
    var x = parm.shape;
    //display the image in a canvas element
    tf.browser.toPixels(parm, document.getElementsByTagName('canvas')[1]);
}

function predictOriginal(model, preprocessed_image) {
    result = model.predict(preprocessed_image);
    return result;
}

function displayAdversarialImage(adv_image) {
    adv_image = tf.image.resizeNearestNeighbor(adv_image, [300, 300]);
    adv_image = tf.squeeze(adv_image);
    var x = adv_image.shape;
    //display the image in a canvas element
    tf.browser.toPixels(adv_image, document.getElementsByTagName("canvas")[2]);
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

function getGradient(img, yTrue) {
    function f(img) {
        yTrue = tf.oneHot(tf.tensor1d([yTrue], 'int32'), 10); //true label of the image
        return tf.metrics.categoricalCrossentropy(yTrue, model.predict(img)); //loss function    
    }
    var g = tf.grad(f); //gradient of loss function
    return g(img); //gradient of loss function w.r.t image
}

function generateAdversarialImage(image,epsilon,pertubation) {
    epsilon = tf.tensor1d([epsilon], 'float32');
    var x = tf.sign(pertubation).mul(epsilon); //pertubation from getGradient function
    x = image.add(x).clipByValue(0,1);
    return x;
}
