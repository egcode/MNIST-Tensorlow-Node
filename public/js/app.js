 
animSpeed = 100;

function uiState(state) {
    if (state == ENUM_INITIAL_STATE) {
        $("#CNNContainer").hide(animSpeed);
        $("#guess-digit").show(animSpeed);
        $("#clear").show(animSpeed);
    } else if (state == ENUM_PROCESS_STATE) {
        $("#CNNContainer").hide(animSpeed);
        $("#guess-digit").hide(animSpeed);
        $("#clear").hide(animSpeed);
    } else if (state == ENUM_COMPLETED_STATE) {
        $("#CNNContainer").show(animSpeed);
        $("#guess-digit").show(animSpeed);
        $("#clear").show(animSpeed);
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

    var target = document.getElementById('topContainer');
    var spinner = new Spinner(opts).spin(target);

    uiState(ENUM_PROCESS_STATE);

    $.ajax('http://localhost:3030/process', {
        type: 'POST',
        data: JSON.stringify({"imageData": imageBase64}),
        contentType: 'application/json',
        success: function(responseFromPython) { 
            console.log('\n\nCLIENT: success responseFromPython:');
            console.log(responseFromPython);

            // PREDICTION
            var prediction = extractPredictionFromResponse(responseFromPython);
            console.log("\nPrediction: " + prediction[1]);
            
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
            assignDenseAndSoftmaxValues(dense, prediction);
            setTimeout(function(){
                updateCNNLinesPosition();
                updateDenseLinesPosition();
                updateSoftmaxLinesPosition();
                updateReultLinesPosition();
            }, 100);
              

            // SOFTMAX
            assignSoftmaxValues(prediction);
              
            // RESULT
            $('#resultNeuron').text(prediction[1]);
            $('#resultNeuron').css("color", "red"); // Font color        

            uiState(ENUM_COMPLETED_STATE);
            spinner.stop(target);

            $('html, body').animate({ 
                scrollTop: $(document).height()-($(window).height() - 1000)}, 1400);
             
                     
        },
        error  : function(data) { 
            uiState(ENUM_INITIAL_STATE);
            console.log('CLIENT: error');
            console.log(data);
            spinner.stop(target);

        }
    }); 
}

function assignSoftmaxValues(prediction) {
    var prediction_int = parseInt(prediction[1]);
    // SOFTMAX Layer
    var softmaxId = "#softmax" + (prediction_int);
    $(softmaxId).css({
        'background-color': 'black'
    });
    // SOFTMAX Recolor Lines
    var softmaxLineId = "#result_softmax_" + (prediction_int) + "_line";
    // console.log("line: " + line);
    $(softmaxLineId).css({
        opacity: 0.9
    });
}

function assignDenseAndSoftmaxValues(dense, prediction){
    var denseJson = JSON.parse(dense);
    for (var key in denseJson) {
        if (denseJson.hasOwnProperty(key)) {
            var value = denseJson[key].split("tuple");
            var denseIndex = value[0];
            var denseValue = value[1];
            var divId = "#dense" + denseIndex;

            // Neuron color
            var colorV = (255.0 * (1-denseValue));
            // console.log("denseValue: " + denseValue);
            // console.log("colorV: " + colorV);
            $(divId).css({
                'background-color':"rgb(" + colorV + "," + colorV + "," + colorV + ")"
            });

            // DENSE Recolor Lines
            var lineId = "#line" + (denseIndex);
            // console.log("line: " + line);
            $(lineId).css({
                opacity: 0.9
            });

            // SOFTMAX Recolor Lines
            var prediction_int = parseInt(prediction[1]);
            // var softmaxId = "#softmax" + (prediction_int-1);
            var lineSoftmaxId = "#softmax_" + (prediction_int) + "_line_" + (denseIndex);
            // console.log("aaa\n\n\n");
            // console.log("CORRECT denseIndex: " + denseIndex);
            // console.log("prediction_int: " + prediction_int);
            // console.log("softmaxId: " + softmaxId);
            // console.log("lineSoftmaxId: " + lineSoftmaxId);

            $(lineSoftmaxId).css({
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

  // --------------- RESIZE METHODS ----------------------

  function updateCNNLinesPosition() {
        // ADJUST CNN Lines
        var lineId1 = "cnn_" + 1 + "_line";
        var lineId2 = "cnn_" + 2 + "_line";
        var lineId3 = "cnn_" + 3 + "_line";
        if ( $("#" + lineId1 ).length && $("#" + lineId2).length && $("#" + lineId3).length){
            adjustLine(
                document.getElementById("topContainer"), 
                document.getElementById("layer1"),
                document.getElementById(lineId1)
                );
                adjustLine(
                document.getElementById("layer1"), 
                document.getElementById("layer2"),
                document.getElementById(lineId2)
                );
                adjustLine(
                document.getElementById("layer2"), 
                document.getElementById("layer3"),
                document.getElementById(lineId3)
                );
        }
  }

  function updateReultLinesPosition() {
    // AJUST Result Lines
    for (i = 1; i <= 10; i++) { 
        var softmaxId = "softmax" + (i-1);
  
        // Add Line
        var lineId = "result_softmax_" + (i-1) + "_line";
  
        if ($("#" + lineId).length && $("#" + softmaxId).length){
            adjustLine(
            document.getElementById(softmaxId), 
            document.getElementById("resultNeuron"),
            document.getElementById(lineId)
          );
        }
    }
  }
  function updateSoftmaxLinesPosition(){
    // ADJUST Softmax Lines
    for (j = 1; j <= 10; j++) { 
      for (i = 1; i <= 200; i++) { 
        var softmaxId = "softmax" + (j-1);
        var denseId = "dense" + (i-1);
        var lineId = "softmax_" + (j-1) + "_line_" + (i-1);
  
        if ( $("#" + denseId ).length && $("#" + lineId).length && $("#" + softmaxId).length){
            adjustLine(
            document.getElementById(softmaxId), 
            document.getElementById(denseId),
            document.getElementById(lineId)
          );
        }
      }
    }
  }
  function updateDenseLinesPosition(){
      // ADJUST Dense Lines
      for (i = 1; i <= 200; i++) { 
        var denseId = "dense" + (i-1);
        var lineId = "line" + (i-1);
  
        if ( $("#" + denseId ).length && $("#" + lineId).length) {
            adjustLine(
            document.getElementById('layer3'), 
            document.getElementById(denseId),
            document.getElementById(lineId)
          );
        }
      }
  }
  

    // --------------- INIT COLORS ----------------------

    function initColors(){
        uiState(ENUM_INITIAL_STATE);

        // INIT Dense 
        for (i = 1; i <= 200; i++) { 
            var denseId = "#dense" + (i-1);
            var lineId = "#line" + (i-1);

            // Neuron color
            $(denseId).css({
                'background-color':"white"
            });

            // DENSE Recolor Lines
            $(lineId).css({
                opacity: 0.1
            });

            // INIT Softmax Lines
            for (j = 1; j <= 10; j++) { 
                // INIT Dense - Softmax Line
                var lineId = "#softmax_" + (j-1) + "_line_" + (i-1);    
                $(lineId).css({
                    opacity: 0.1
                });
            }
        }
        // INIT Softmax Neurons
        for (j = 1; j <= 10; j++) { 
            var softmaxId = "#softmax" + (j-1);
            $(softmaxId).css({
                'background-color':"white"
            });

            // INIT Result Lines
            var lineId = "#result_softmax_" + (j-1) + "_line";
            $(lineId).css({
                opacity: 0.1
            });
        }
        // INIT RESULT
        $('#resultNeuron').css("color", "rgba(0, 0, 0, 0.0)"); // Font color        
}


