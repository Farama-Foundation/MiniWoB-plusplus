// Hacks to make JQuery UI deterministic

$(function () {
  var uuid = 0;

  $.extend({
    resetUniqueId: function () {
      uuid = 0;
    },
  });

  $.fn.extend({
    uniqueId: function () {
      return this.each(function () {
        if (!this.id) {
          this.id = "ui-id-" + ( ++uuid );
        }
      });
    },
  });

});
