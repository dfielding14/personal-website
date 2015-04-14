/**
 * Created by lb on 4/13/15.
 */
$(document).ready(function() {
    // initializing page transition.
    PageTransitions.init();

    $("body").keydown(function(e) {
           if(e.keyCode == 37) { // left
               if ($('.left').css('visibility') == 'visible') {
                   $(".left").trigger("click");
               }
           } else if(e.keyCode == 39) { // right
                $(".right").trigger("click");
           } else if(e.keyCode == 40) { // down
                $(".down").trigger("click");
           } else if(e.keyCode == 38) { // top
                $(".top").trigger("click")
           }
      });
});