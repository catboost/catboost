if (window.require) {
    window.require.config({
        map: {
            '*': {
                'catboost-widget': 'nbextensions/catboost-widget/index',
            }
        }
    });
}

module.exports = {
    load_ipython_extension: function () {
    }
};
