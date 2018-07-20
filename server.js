var express = require('express');    //Express Web Server 
var path = require('path');     //used for file path
var bodyParser = require('body-parser')

var app = express();
app.use(express.static(path.join(__dirname, 'public')));

// parse application/x-www-form-urlencoded
// app.use(bodyParser.urlencoded({ extended: false }))
 
// parse application/json
app.use(bodyParser.json())

var server = app.listen(3030, function() {
    console.log('Listening on port %d', server.address().port);
});

///////////////////////////////////
// PYTHON
///////////////////////////////////

app.post('/process', callName);
 
function callName(req, res) {
     
    // var sys = require('util');
    var imageBase64Data = req.body["imageData"]
    var projectPath = __dirname;  // Users/yujin/Desktop/nodeMNIST
    var modelPath = __dirname + "/Python_NN/saved_model/model.ckpt"; // Users/yujin/Desktop/nodeMNIST/Python_NN/saved_model/model.ckpt

    var spawn = require("child_process").spawn;
          
    var process = spawn('python',["./Python_NN/app_cnn_tf_mnist.py", 
                                        imageBase64Data.toString(),
                                        modelPath] );
 
    // Takes stdout data from script which executed
    // with arguments and send this data to res object
    // process.stdout.on('data', function(data) {
        // console.log("\n\nResponse from python: " + data.toString());

        // var retryButton = "<form action=\"/\" method=\"get\"><button>Retry</button></form>";
        // var comment = "<img src=\"/img/image.png\" width=\"500\"><div class=\"messages\">" + data.toString() + "</div>";
        // comment = comment + retryButton;
        // res.send(comment);

        // res.send(data.toString());

        // res.status(200).json({ "imData": data.toString() })
        
    // });

    str = "";

    process.stdout.on('data', function (data) {
        str += data.toString();

        // just so we can see the server is doing something
        console.log("data");

        // Flush out line by line.
        var lines = str.split("\n");
        for(var i in lines) {
            if(i == lines.length - 1) {
                str = lines[i];
            } else{
                // Note: The double-newline is *required*
                res.write('data: ' + lines[i] + "\n\n");
            }
        }
    });

    process.on('close', function (code) {
        res.end();
    });

    process.stderr.on('data', function (data) {
        res.end('stderr: ' + data);
    });
    
}





