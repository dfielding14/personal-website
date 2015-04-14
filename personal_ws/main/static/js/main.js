/**
 * Created by lb on 4/13/15.
 */
$(document).ready(function() {
    // initializing page transition.
    PageTransitions.init();

    // Arrow Keys
    var directionMap = {37: '.left', 39: '.right', 40: '.down', 38: '.top'};
    $("body").keydown(function(e) {
        if (e.keyCode <= 40 && e.keyCode >= 37) {
            var direction = directionMap[e.keyCode];
            if ($(direction).css('visibility') == 'visible') {
                $(direction).trigger('click');
            }
        }
    });
});