

$(document).ready(function() {
    $('#show-same-stacks').click(function() {
        var stacks = $('.same-hash');
        for (var i = 0; i < stacks.length; ++i)
            $(stacks[i]).show();
        $('#show-same-stacks').hide();
        $('#hide-same-stacks').show();
        return false;
    });

    $('#hide-same-stacks').click(function() {
        var stacks = $('.same-hash');
        for (var i = 0; i < stacks.length; ++i)
            $(stacks[i]).hide();
        $('#hide-same-stacks').hide();
        $('#show-same-stacks').show();
        return false;
    });
});
