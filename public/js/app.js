 
animSpeed = 100;

function uiState(state) {
    if (state == ENUM_INITIAL_STATE) {
        // $("#formDiv").show(animSpeed);
        // $("#imageDiv").hide(animSpeed);
        // $("#captionDiv").hide(animSpeed);
        // $(".messages").hide(animSpeed);
        // $("#refreshButtonDiv").hide(animSpeed);
    } else if (state == ENUM_PREPROCESS_STATE) {
        // $("#formDiv").hide(animSpeed);
        // $("#imageView").attr('src', '/img/imagedata');
        // $("#imageDiv").show(animSpeed);
        // $("#captionDiv").show(animSpeed);
        // $("#refreshButtonDiv").hide(animSpeed);
    } else if (state == ENUM_COMPLETED_STATE) {
        // uiState(ENUM_PREPROCESS_STATE);
        // $("#refreshButtonDiv").show(animSpeed);
    }
  }

// function uploadImage() {
//     $('#uploadForm').submit(function() {

//         var target = document.getElementById('topContainer');
//         var spinner = new Spinner(opts).spin(target);

//         $("#status").empty().text("File is uploading...");
//         $(this).ajaxSubmit({

//         error: function(xhr) {
//             status('Error: ' + xhr.status);
//             spinner.stop(target);
//         },

//         success: function(response) {
//             $("#status").empty().text(response);
//                 console.log(response);
//                 console.log("ZAYEBIS UPLOADED");
//                 uiState(ENUM_PREPROCESS_STATE);
//                 spinner.stop(target);
//         }
//     });
//         //Very important line, it disable the page refresh.
//         return false;
//     });    
// }
  
function getMnistPredictionFromServer(imageBase64) {
    console.log("Zayebis getMnistPredictionFromServer");
    console.log(imageBase64);
    // Hide button
    // $("#caption").hide(animSpeed);

    // var target = document.getElementById('imageDiv');
    // var spinner = new Spinner(opts).spin(target);

    $.ajax('http://localhost:3030/process', {
        type: 'POST',
        data: JSON.stringify({"imageData": imageBase64}),
        contentType: 'application/json',
        success: function(responseFromPython) { 
            console.log('\n\nCLIENT: success responseFromPython:');
            console.log(responseFromPython);

            // CNN Layer1 
            for (i = 1; i <= 6; i++) { 
                var id = "#L1_i" + i + "_imageView";
                $(id).attr('src', extractImageFromResponse(responseFromPython,1,i));
            }

            // CNN Layer2 
            for (i = 1; i <= 12; i++) { 
                var id2 = "#L2_i" + i + "_imageView";
                $(id2).attr('src', extractImageFromResponse(responseFromPython,2,i));
            }
            
            // CNN Layer3 
            for (i = 1; i <= 24; i++) { 
                var id3 = "#L3_i" + i + "_imageView";
                $(id3).attr('src', extractImageFromResponse(responseFromPython,3,i));
            }
            
            // DENSE Layer
            var dense = extractDenseFromResponse(responseFromPython);
            // console.log("\nDense: " + dense);
            assignDenseValues(dense);
            setTimeout(function(){
                updateDenseLinesPosition();
                updateSoftmaxLinesPosition();
                updateReultLinesPosition();
            }, 100);
              
            // PREDICTION
            var prediction = extractPredictionFromResponse(responseFromPython);
            console.log("\nPrediction: " + prediction[1]);
            var prediction_int = parseInt(prediction[1]);

            
            // SOFTMAX Layer
            var softmaxId = "#softmax" + (prediction_int-1);
            $(softmaxId).css({
                'background-color': 'black'
            });
            // SOFTMAX Recolor Lines
            var softmaxLineId = "#result_softmax_" + (prediction_int-1) + "_line";
            // console.log("line: " + line);
            $(softmaxLineId).css({
                opacity: 0.9
            });
            
            
            // var $message = jQuery('.messages');//getting text from textField
            // $message.append('<h1><strong>' + caption + '</strong></h1>');
            // $(".messages").show(animSpeed);

            // uiState(ENUM_COMPLETED_STATE);
            // spinner.stop(target);

        },
        error  : function(data) { 
            // uiState(ENUM_INITIAL_STATE);
            console.log('CLIENT: error');
            console.log(data);
            // spinner.stop(target);

        }
    }); 
}

function assignDenseValues(dense){
    var denseJson = JSON.parse(dense);
    for (var key in denseJson) {
        if (denseJson.hasOwnProperty(key)) {
            var value = denseJson[key].split("tuple");
            var denseIndex = value[0];
            var denseValue = value[1];
            // console.log("dense Index: " + denseIndex + " dense Value: " + denseValue);
            var divId = "#dense" + denseIndex;
            // console.log("divId: " + divId);

            // Neuron color
            var colorV = (255.0 * (1-denseValue));
            console.log("denseValue: " + denseValue);
            console.log("colorV: " + colorV);
            $(divId).css({
                // opacity: denseValue,
                'background-color':"rgb(" + colorV + "," + colorV + "," + colorV + ")"
                // 'border': '2px solid black'
            });

            // ADD Recolor Lines
            var lineId = "#line" + (denseIndex);
            // console.log("line: " + line);
            $(lineId).css({
                opacity: 0.9
            });

        }
    }
}

function extractPredictionFromResponse(source_string) {
    var start = "Predicted_number--->";
    var end = "<---Predicted_number";
    return source_string.substring(source_string.indexOf(start) + start.length-1, source_string.indexOf(end));
}

function extractDenseFromResponse(source_string) {
    var start = "dense_layer--->";
    var end = "<---dense_layer";
    return source_string.substring(source_string.indexOf(start) + start.length, source_string.indexOf(end));
}


function extractImageFromResponse(source_string, layer_num, image_num) {
    var im_prefix = "data:image/jpeg;base64,";
    var start = "base64_layer" + layer_num + "_image" + image_num + "--->";
    var end = "<---base64_layer" + layer_num + "_image" + image_num + "_";
    // console.log("layer: " + layer_num + " image: " + image_num);
    var image_arr = source_string.match(new RegExp(start + "(.*)" + end));
    if (image_arr[1] != null) {
        return im_prefix + image_arr[1];
    } else {
        return "";
    }
}

function refreshClick() {
    location.reload(); // reload all
}


function adjustLine(from, to, line){

    var fT = from.offsetTop  + from.offsetHeight/2;
    var tT = to.offsetTop    + to.offsetHeight/2;
    var fL = from.offsetLeft + from.offsetWidth/2;
    var tL = to.offsetLeft   + to.offsetWidth/2;
    
    var CA   = Math.abs(tT - fT);
    var CO   = Math.abs(tL - fL);
    var H    = Math.sqrt(CA*CA + CO*CO);
    var ANG  = 180 / Math.PI * Math.acos( CA/H );
  
    if(tT > fT){
        var top  = (tT-fT)/2 + fT;
    }else{
        var top  = (fT-tT)/2 + tT;
    }
    if(tL > fL){
        var left = (tL-fL)/2 + fL;
    }else{
        var left = (fL-tL)/2 + tL;
    }
  
    if(( fT < tT && fL < tL) || ( tT < fT && tL < fL) || (fT > tT && fL > tL) || (tT > fT && tL > fL)){
      ANG *= -1;
    }
    top-= H/2;
  
    line.style["-webkit-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-moz-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-ms-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-o-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-transform"] = 'rotate('+ ANG +'deg)';
    line.style.top    = top+'px';
    line.style.left   = left+'px';
    line.style.height = H + 'px';
  }