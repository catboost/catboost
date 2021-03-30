if (window.require) {
    window.require.config({
        map: {
            '*': {
                'catboost_widget': 'nbextensions/catboost_widget/index',
            }
        }
    });
}

module.exports = {
    load_ipython_extension: function () {
    }
};
