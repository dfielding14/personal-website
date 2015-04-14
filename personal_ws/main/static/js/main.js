/**
 * Created by lb on 4/13/15.
 */
$(document).ready(function() {
    // initializing page transition.
    PageTransitions.init();

    // Arrow Keys
    var directionMap = {37: '.left', 39: '.right', 40: '.down', 38: '.top'};
    var inQueue = false;

    $("body").keydown(function(e) {
        if (!inQueue) {
            if (e.keyCode <= 40 && e.keyCode >= 37) {
                var direction = directionMap[e.keyCode];
                var availableTriggers = [];
                $('.pt-trigger').each(function(){
                    if ($(this).css('visibility') == 'visible') {
                        availableTriggers.push(this.className.split(' ')[0])
                    }
                });
                var transition = false;
                for (var i = 0; i < availableTriggers.length; i++) {
                    if ("." + availableTriggers[i] == direction) {
                        transition = true;
                    }
                }
                if (transition) {
                    inQueue = true;
                    $(direction).trigger('click');
                    setTimeout(function(){ inQueue = false}, 1000);
                }
            }
        }
    });
});