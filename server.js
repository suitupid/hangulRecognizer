let express = require("express");
let app = express();
let http = require("http");
let server = http.createServer(app);
let io = require("socket.io")(server);
let fs = require("fs");

app.use("/css", express.static("css"));
app.use("/js", express.static("js"));

app.get("/", function(req, res) {
    res.sendFile(__dirname + "/index.html");
});

server.listen(3000, function() {
    console.log("Listening on port 3000");
});

io.on("connection", function(socket) {
	address = socket.request.connection.remoteAddress;
	console.log("Connected: "+socket.id+"("+address+")");
	socket.on("request", (reqData)=>{
		console.log("Get Request: "+socket.id+"("+address+")");
		matches = reqData.match(/^data:.+\/(.+);base64,(.*)$/);
		fs.writeFileSync(
			"./python/image/"+socket.id+"."+matches[1],
			Buffer.from(matches[2], "base64")
		);
        url = 'http://127.0.0.1:8000/predict/'+socket.id+"."+matches[1];
        http.get(url, function(resp) {
            let respData = '';
            resp.on('data', function(chunk) {
                respData += chunk
            });
            resp.on('end', function() {
                respData = respData.replace(/"/g, '');
                io.emit("response_"+socket.id, respData.toString());
                console.log("Responsed: "+socket.id+"("+address+")");
            });
        });
	});
});
