$(document).ready(function(){
  var mediaOptions = { audio: false, video: true };

  if (!navigator.getUserMedia) {
      navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
  }

  if (!navigator.getUserMedia){
    return alert('getUserMedia not supported in this browser.');
  }

  navigator.getUserMedia(mediaOptions, success, function(e) {
    console.log(e);
  });

  setInterval(snapShot, 1000);

  function success(stream){
    var video = document.querySelector("#player");
    video.src = window.URL.createObjectURL(stream);
  }

  function snapShot(){

    var canvas = document.querySelector("#canvas"),
        video = document.querySelector("#player");

    canvas.width = 224;
    canvas.height = 224;

    var context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, 224, 224)

    var post_data = {image: canvas.toDataURL('image/jpg')}
    $.post('/api/score', post_data, function(data){
      console.log(data)
      var vgg = data.vgg

      var top_tag = _.chain(vgg)
        .sortBy(function(x){return -x.score})
        .first()
        .value();

      $("#label").text(top_tag.tag)
    })
  }
});
